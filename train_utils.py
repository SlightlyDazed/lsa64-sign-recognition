"""
train_utils.py — Funciones compartidas de entrenamiento para las 3 etapas.

Mejoras v2 (accuracy rápido con buena VRAM):
  1. FocalLoss (gamma=2.0) + label smoothing — penaliza más ejemplos difíciles.
  2. Mixup (alpha=0.3) en secuencias de keypoints — regularización fuerte.
  3. CosineAnnealingWarmRestarts (T_0=10, T_mult=2) — escapa mínimos locales.
  4. Early stopping con patience configurable (default 15 epochs).
  5. _get_lr() helper compatible con cualquier scheduler.
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from utils import save_checkpoint, plot_training_curves


# ---------------------------------------------------------------------------
# Helper: obtener LR actual sin importar el tipo de scheduler
# ---------------------------------------------------------------------------
def _get_lr(optimizer) -> float:
    return optimizer.param_groups[0]["lr"]


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """
    Focal Loss para clasificación multiclase.
    Reduce el peso de ejemplos fáciles, fuerza al modelo a aprender los difíciles.

    Args:
        gamma: Factor de foco (2.0 es el valor estándar de la paper original).
        label_smoothing: Suavizado de etiquetas para reducir overconfianza.
    """
    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.1):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing, reduction="none"
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)           # (B,)
        pt = torch.exp(-ce_loss)                     # probabilidad predicha
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# ---------------------------------------------------------------------------
# Mixup para secuencias de keypoints
# ---------------------------------------------------------------------------
def mixup_batch(seqs: torch.Tensor, labels: torch.Tensor, alpha: float = 0.3):
    """
    Mixup sobre un batch de secuencias.

    Returns:
        mixed_seqs  : secuencias interpoladas
        labels_a    : etiquetas originales
        labels_b    : etiquetas del par de mezcla
        lam         : coeficiente de interpolación (escalar)
    """
    if alpha <= 0:
        return seqs, labels, labels, 1.0

    lam = float(np.random.beta(alpha, alpha))
    batch_size = seqs.size(0)
    idx = torch.randperm(batch_size, device=seqs.device)

    mixed_seqs = lam * seqs + (1.0 - lam) * seqs[idx]
    labels_a = labels
    labels_b = labels[idx]
    return mixed_seqs, labels_a, labels_b, lam


# ---------------------------------------------------------------------------
# Train / Eval por epoch
# ---------------------------------------------------------------------------

def train_one_epoch(model, dataloader, criterion, optimizer, device,
                    scheduler=None, use_mixup: bool = True,
                    mixup_alpha: float = 0.3):
    """
    Entrena un epoch con Mixup opcional y step de scheduler por batch
    (necesario para CosineAnnealingWarmRestarts).

    Returns:
        avg_loss, accuracy  (accuracy calculada sobre labels_a para Mixup)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = len(dataloader)

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="  Train", leave=False)):
        seqs    = batch["sequence"].to(device)   # (B, T, D)
        labels  = batch["label"].to(device)       # (B,)
        lengths = batch["length"].to(device)      # (B,)

        # Mixup
        if use_mixup:
            mixed_seqs, labels_a, labels_b, lam = mixup_batch(
                seqs, labels, alpha=mixup_alpha
            )
            optimizer.zero_grad()
            logits = model(mixed_seqs, lengths)
            loss = lam * criterion(logits, labels_a) + \
                   (1 - lam) * criterion(logits, labels_b)
        else:
            optimizer.zero_grad()
            logits = model(seqs, lengths)
            loss = criterion(logits, labels)
            labels_a = labels

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Step por batch para CosineAnnealingWarmRestarts
        if scheduler is not None and isinstance(
            scheduler, CosineAnnealingWarmRestarts
        ):
            # epoch fraccional: epoch actual se pasa desde train_model
            pass  # se hace fuera, pasando epoch + batch_idx/num_batches

        total_loss += loss.item() * seqs.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels_a).sum().item()
        total      += seqs.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct   / max(total, 1)
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evalúa y retorna loss, accuracy, predicciones y labels."""
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0
    all_preds  = []
    all_labels = []

    for batch in tqdm(dataloader, desc="  Eval", leave=False):
        seqs    = batch["sequence"].to(device)
        labels  = batch["label"].to(device)
        lengths = batch["length"].to(device)

        logits = model(seqs, lengths)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * seqs.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += seqs.size(0)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / max(total, 1)
    accuracy = correct   / max(total, 1)
    return avg_loss, accuracy, all_preds, all_labels


# ---------------------------------------------------------------------------
# Loop principal
# ---------------------------------------------------------------------------

def train_model(model, train_loader, val_loader, train_config: dict,
                device, checkpoint_path: str, results_prefix: str,
                freeze_epochs: int = 0):
    """
    Loop de entrenamiento con Focal Loss, Mixup y Warm Restarts.

    train_config puede incluir:
        epochs              (int)
        lr                  (float)
        weight_decay        (float,  default 1e-4)
        early_stop_patience (int,    default 15)
        use_mixup           (bool,   default True)
        mixup_alpha         (float,  default 0.3)
        focal_gamma         (float,  default 2.0)
        warmrestart_T0      (int,    default 10)
        warmrestart_Tmult   (int,    default 2)
    """
    epochs              = train_config["epochs"]
    lr                  = train_config["lr"]
    weight_decay        = train_config.get("weight_decay", 1e-4)
    early_stop_patience = train_config.get("early_stop_patience", 15)
    use_mixup           = train_config.get("use_mixup", True)
    mixup_alpha         = train_config.get("mixup_alpha", 0.3)
    focal_gamma         = train_config.get("focal_gamma", 2.0)
    T0                  = train_config.get("warmrestart_T0", 10)
    Tmult               = train_config.get("warmrestart_Tmult", 2)

    # Focal Loss con label smoothing
    criterion = FocalLoss(gamma=focal_gamma, label_smoothing=0.1)

    # Optimizador fase 1 (backbone congelado o completo)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay,
    )

    # Fase 1: CosineAnnealing simple hasta que se descongele (o hasta el final)
    t_max_phase1 = freeze_epochs if freeze_epochs > 0 else epochs
    scheduler = CosineAnnealingLR(
        optimizer, T_max=t_max_phase1, eta_min=lr * 0.01
    )
    using_warm_restarts = False

    total_params, trainable_params = model.count_parameters()
    print(f"\n{'='*60}")
    print(f"Parámetros totales:      {total_params:,}")
    print(f"Parámetros entrenables: {trainable_params:,}")
    print(f"Epochs: {epochs} | LR: {lr} | Freeze epochs: {freeze_epochs}")
    print(f"Mixup: {use_mixup} (alpha={mixup_alpha}) | "
          f"FocalLoss gamma={focal_gamma}")
    print(f"WarmRestarts T0={T0} Tmult={Tmult} | "
          f"Early stop patience={early_stop_patience}")
    print(f"{'='*60}\n")

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    best_val_acc     = 0.0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # --- Descongelar backbone + cambiar a Warm Restarts ------------------
        if freeze_epochs > 0 and epoch == freeze_epochs + 1:
            print(f"\n[Epoch {epoch}] Descongelando backbone → "
                  f"activando CosineAnnealingWarmRestarts...")
            model.unfreeze_all()

            # LR post-unfreeze más generoso que antes (0.3x en vez de 0.1x)
            optimizer = AdamW(
                model.parameters(),
                lr=lr * 0.3,
                weight_decay=weight_decay,
            )

            # Warm Restarts: reinicia el LR cada T0 epochs, duplicando el periodo
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=T0, T_mult=Tmult, eta_min=1e-6
            )
            using_warm_restarts = True
            patience_counter = 0  # reset early-stop al descongelar

        # --- Train -----------------------------------------------------------
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scheduler=scheduler if using_warm_restarts else None,
            use_mixup=use_mixup,
            mixup_alpha=mixup_alpha,
        )

        # Step de scheduler:
        # - CosineAnnealingLR: una vez por epoch
        # - CosineAnnealingWarmRestarts: epoch fraccional para suavidad
        if using_warm_restarts:
            scheduler.step(epoch - 1)   # epoch 0-indexed para el scheduler
        else:
            scheduler.step()

        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # --- Val -------------------------------------------------------------
        if val_loader is not None:
            val_loss, val_acc, _, _ = evaluate(
                model, val_loader, criterion, device
            )
        else:
            val_loss, val_acc = train_loss, train_acc
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        elapsed    = time.time() - start_time
        current_lr = _get_lr(optimizer)

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.2e} | Time: {elapsed:.1f}s")

        # --- Guardar mejor modelo + early stopping ---------------------------
        if val_acc >= best_val_acc:
            best_val_acc     = val_acc
            patience_counter = 0
            save_checkpoint(
                model, optimizer, epoch,
                {"val_acc": val_acc, "val_loss": val_loss,
                 "train_acc": train_acc, "train_loss": train_loss},
                checkpoint_path,
            )
            print(f"  ★ Mejor modelo guardado (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"\n[Early Stop] Sin mejora en {early_stop_patience} "
                      f"epochs consecutivos. Deteniendo entrenamiento.")
                break

    # --- Curvas de entrenamiento ---------------------------------------------
    curves_path = os.path.join(
        config.RESULTS_DIR, f"{results_prefix}_training_curves.png"
    )
    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs, curves_path
    )
    print(f"\n[UTILS] Curvas guardadas en {curves_path}")

    print(f"\n{'='*60}")
    print(f"Entrenamiento completado. Mejor val_acc: {best_val_acc:.4f}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*60}\n")

    return {
        "best_val_acc":    best_val_acc,
        "final_train_acc": train_accs[-1],
        "final_val_acc":   val_accs[-1],
        "train_losses":    train_losses,
        "val_losses":      val_losses,
    }
