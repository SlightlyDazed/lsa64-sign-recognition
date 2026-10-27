"""
model.py — Clasificador temporal de secuencias de keypoints para señas.
Arquitectura: Linear projection → BiLSTM → Classification head
"""
import torch
import torch.nn as nn


class SignClassifier(nn.Module):
    """
    Clasificador de señas basado en secuencias temporales de keypoints.

    Arquitectura:
        1. input_proj: Proyección lineal de keypoints aplanados a espacio latente
        2. temporal: LSTM bidireccional para capturar dinámica temporal
        3. head: Capa de clasificación

    Transfer learning:
        - Para cambiar de dataset, reemplazar `head` con replace_head()
        - Congelar capas con freeze_backbone() / unfreeze_all()
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 num_layers: int = 2, num_classes: int = 100,
                 dropout: float = 0.3, bidirectional: bool = True):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Proyección de entrada
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Encoder temporal
        # NOTA: dropout=0.0 aquí es intencional.
        # El dropout interno del LSTM usa un kernel HIP (MIOpenDropoutHIP.cpp)
        # que falla al compilar en gfx1201 (RX 9070) por header faltante:
        # 'rocrand/rocrand_xorwow.h'. Solución: aplicar dropout manualmente
        # después del LSTM con un nn.Dropout estándar de PyTorch.
        self.temporal = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0,          # ← 0.0 para evitar crash MIOpen en ROCm gfx1201
            bidirectional=bidirectional,
        )
        # Dropout manual post-LSTM (mismo efecto, sin pasar por MIOpen)
        self.lstm_dropout = nn.Dropout(dropout)

        # Attention pooling sobre la dimensión temporal
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * self.num_directions, 1),
            nn.Softmax(dim=1),
        )

        # Clasificador
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * self.num_directions, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Inicialización Xavier para capas lineales."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch, T, input_dim) — secuencia de keypoints aplanados
            lengths: (batch,) — longitudes reales (para padding mask)

        Returns:
            (batch, num_classes) — logits
        """
        batch_size, T, _ = x.shape

        # Proyección de entrada
        projected = self.input_proj(x)  # (B, T, hidden_dim)

        # LSTM
        if lengths is not None:
            # Pack padded sequences para eficiencia
            packed = nn.utils.rnn.pack_padded_sequence(
                projected, lengths.cpu().clamp(min=1),
                batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.temporal(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True, total_length=T
            )
        else:
            lstm_out, _ = self.temporal(projected)
        lstm_out = self.lstm_dropout(lstm_out)  # dropout manual post-LSTM
        # lstm_out: (B, T, hidden_dim * num_directions)

        # Attention pooling
        attn_weights = self.attention(lstm_out)  # (B, T, 1)
        context = (lstm_out * attn_weights).sum(dim=1)  # (B, hidden_dim * dirs)

        # Clasificación
        logits = self.head(context)  # (B, num_classes)
        return logits

    def replace_head(self, new_num_classes: int):
        """
        Reemplaza la cabeza de clasificación para un nuevo dataset.
        Mantiene input_proj y temporal intactos (transfer learning).
        """
        hidden_dim = self.hidden_dim
        num_dirs = self.num_directions
        dropout = 0.3  # mantener mismo dropout

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * num_dirs, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, new_num_classes),
        )
        # Re-inicializar la nueva head
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        print(f"[MODEL] Head reemplazada: → {new_num_classes} clases")

    def freeze_backbone(self, freeze_input_proj=True, freeze_temporal=True):
        """Congela capas del backbone para fine-tuning."""
        if freeze_input_proj:
            for param in self.input_proj.parameters():
                param.requires_grad = False
            print("[MODEL] input_proj congelado")

        if freeze_temporal:
            for param in self.temporal.parameters():
                param.requires_grad = False
            for param in self.attention.parameters():
                param.requires_grad = False
            print("[MODEL] temporal + attention congelado")

    def unfreeze_all(self):
        """Descongela todas las capas."""
        for param in self.parameters():
            param.requires_grad = True
        print("[MODEL] Todas las capas descongeladas")

    def count_parameters(self):
        """Cuenta parámetros totales y entrenables."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


class SignClassifierTransformer(nn.Module):
    """
    Alternativa con Transformer encoder (opcional).
    Mejor para secuencias largas pero más costosa en memoria.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 num_heads: int = 4, num_layers: int = 4,
                 num_classes: int = 100, dropout: float = 0.3,
                 max_seq_len: int = 64):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_seq_len, hidden_dim) * 0.02
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # CLS token + head
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, lengths=None) -> torch.Tensor:
        batch_size, T, _ = x.shape
        projected = self.input_proj(x)

        # Add positional encoding
        projected = projected + self.pos_encoding[:, :T, :]

        # Prepend CLS token
        cls = self.cls_token.expand(batch_size, -1, -1)
        seq = torch.cat([cls, projected], dim=1)  # (B, T+1, D)

        # Transformer
        encoded = self.temporal(seq)

        # Use CLS token output
        cls_output = encoded[:, 0]  # (B, D)
        logits = self.head(cls_output)
        return logits

    def replace_head(self, new_num_classes: int):
        hidden_dim = self.hidden_dim
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, new_num_classes),
        )

    def freeze_backbone(self, freeze_input_proj=True, freeze_temporal=True):
        if freeze_input_proj:
            for p in self.input_proj.parameters():
                p.requires_grad = False
        if freeze_temporal:
            for p in self.temporal.parameters():
                p.requires_grad = False
            self.pos_encoding.requires_grad = False
            self.cls_token.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
