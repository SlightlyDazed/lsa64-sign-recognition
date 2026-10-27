"""
dataset_loader.py — Dataset y DataLoaders para secuencias de keypoints.
Soporta los 3 datasets: WLASL, AUTSL, LSP PUCP.
"""
import os
import json
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from utils import pad_or_truncate


# ============================================================
# Augmentaciones temporales
# ============================================================
class TemporalAugmentor:
    """Augmentaciones para secuencias de keypoints."""

    def __init__(self, enabled: bool = True,
                 temporal_crop_prob: float = 0.3,
                 speed_jitter_prob: float = 0.3,
                 keypoint_jitter_prob: float = 0.5,
                 keypoint_jitter_std: float = 0.01):
        self.enabled = enabled
        self.temporal_crop_prob = temporal_crop_prob
        self.speed_jitter_prob = speed_jitter_prob
        self.keypoint_jitter_prob = keypoint_jitter_prob
        self.keypoint_jitter_std = keypoint_jitter_std

    def __call__(self, seq: np.ndarray) -> np.ndarray:
        """
        Aplica augmentaciones a una secuencia (T, K, 3).
        """
        if not self.enabled:
            return seq

        # Random temporal crop (recortar porción de la secuencia)
        if np.random.random() < self.temporal_crop_prob and len(seq) > 10:
            T = len(seq)
            crop_ratio = np.random.uniform(0.7, 1.0)
            new_len = max(8, int(T * crop_ratio))
            start = np.random.randint(0, T - new_len + 1)
            seq = seq[start:start + new_len]

        # Speed jitter (remuestrear a velocidad ligeramente diferente)
        if np.random.random() < self.speed_jitter_prob and len(seq) > 5:
            speed_factor = np.random.uniform(0.8, 1.2)
            new_len = max(5, int(len(seq) / speed_factor))
            indices = np.linspace(0, len(seq) - 1, new_len, dtype=int)
            seq = seq[indices]

        # Keypoint jitter (ruido gaussiano en coordenadas)
        if np.random.random() < self.keypoint_jitter_prob:
            noise = np.random.normal(0, self.keypoint_jitter_std, seq.shape)
            noise[:, :, 2] = 0  # No perturbar confianza
            # Solo aplicar ruido a keypoints válidos (conf > 0)
            valid = seq[:, :, 2:3] > 0
            noise[:, :, :2] *= valid
            seq = seq + noise

        # Random horizontal flip (espejo — invertir x)
        if np.random.random() < 0.5:
            seq_flipped = seq.copy()
            seq_flipped[:, :, 0] = -seq_flipped[:, :, 0]  # Normalizado asume centro 0, invertir signo

            # Si hay manos (0-20 = left, 21-41 = right)
            if seq.shape[1] == 42:
                left_hand = seq_flipped[:, 0:21].copy()
                right_hand = seq_flipped[:, 21:42].copy()
                seq_flipped[:, 0:21] = right_hand
                seq_flipped[:, 21:42] = left_hand

            seq = seq_flipped

        return seq.astype(np.float32)


# ============================================================
# Dataset genérico de secuencias
# ============================================================
class SignSequenceDataset(Dataset):
    """
    Dataset de secuencias de keypoints para clasificación de señas.

    Espera archivos .npy organizados como:
        base_dir/{split}/{class_name}/video_id.npy

    Cada .npy tiene shape (T, K, 3).
    """

    def __init__(self, data_dir: str, split: str = "train",
                 max_seq_len: int = 64,
                 augmentor: TemporalAugmentor = None,
                 class_to_idx: Dict[str, int] = None,
                 max_samples: int = None):
        # Soportar dos estructuras:
        #   - Clásica (WLASL/AUTSL): data_dir/sequences/{split}/{class}/file.npy
        #   - Flat (LSA64):          data_dir/{split}/NNN_sXX_rXX.npy  (sin subcarpeta de clase)
        split_with_seq = os.path.join(data_dir, "sequences", split)
        split_flat     = os.path.join(data_dir, split)

        if os.path.isdir(split_with_seq):
            self.data_dir  = split_with_seq
            self.flat_mode = False
        elif os.path.isdir(split_flat):
            self.data_dir  = split_flat
            self.flat_mode = True   # LSA64: .npy contiene dict con "sequence" y "label"
        else:
            self.data_dir  = split_with_seq  # path original para el mensaje de error
            self.flat_mode = False

        self.max_seq_len = max_seq_len
        self.augmentor   = augmentor
        self.max_samples = max_samples

        # Descubrir clases y archivos
        self.samples: List[Tuple[str, int]] = []  # (path, label_idx)

        if not os.path.isdir(self.data_dir):
            print(f"[DATASET] Aviso: directorio no encontrado: {self.data_dir}")
            self.class_to_idx = class_to_idx or {}
            self.idx_to_class = {}
            return

        if self.flat_mode:
            # Modo LSA64: archivos planos NNN_sXX_rXX.npy, label en el propio .npy
            if class_to_idx is not None:
                self.class_to_idx = class_to_idx
            else:
                self.class_to_idx = {}

            for npy_file in sorted(os.listdir(self.data_dir)):
                if not npy_file.endswith(".npy"):
                    continue
                path = os.path.join(self.data_dir, npy_file)
                try:
                    data      = np.load(path, allow_pickle=True).item()
                    label_idx = int(data["label"])
                    self.samples.append((path, label_idx))
                except Exception as e:
                    print(f"[DATASET] WARN: no se pudo leer {npy_file}: {e}")

            # Construir class_to_idx desde class_mapping.json si no se pasó
            if not self.class_to_idx:
                mapping_path = os.path.join(
                    os.path.dirname(self.data_dir), "class_mapping.json"
                )
                if os.path.exists(mapping_path):
                    with open(mapping_path) as f:
                        self.class_to_idx = json.load(f)
                else:
                    # Inferir desde labels únicos
                    unique_labels = sorted(set(lbl for _, lbl in self.samples))
                    self.class_to_idx = {str(i): i for i in unique_labels}

        else:
            # Modo clásico: subdirectorios por clase
            classes = sorted([
                d for d in os.listdir(self.data_dir)
                if os.path.isdir(os.path.join(self.data_dir, d))
            ])

            if class_to_idx is not None:
                self.class_to_idx = class_to_idx
            else:
                self.class_to_idx = {c: i for i, c in enumerate(classes)}

            for cls_name in classes:
                if cls_name not in self.class_to_idx:
                    continue
                cls_dir   = os.path.join(self.data_dir, cls_name)
                label_idx = self.class_to_idx[cls_name]
                for npy_file in sorted(os.listdir(cls_dir)):
                    if npy_file.endswith(".npy"):
                        path = os.path.join(cls_dir, npy_file)
                        self.samples.append((path, label_idx))

        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        if max_samples and len(self.samples) > max_samples:
            np.random.seed(42)
            indices = np.random.choice(
                len(self.samples), max_samples, replace=False
            )
            self.samples = [self.samples[i] for i in indices]

        print(f"[DATASET] {split}: {len(self.samples)} muestras, "
              f"{len(self.class_to_idx)} clases "
              f"({'flat/lsa64' if self.flat_mode else 'subdir'} mode)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label_idx = self.samples[idx]

        if self.flat_mode:
            # LSA64 legacy flat: .npy es un dict con "sequence"
            data     = np.load(path, allow_pickle=True).item()
            seq_flat = data["sequence"].astype(np.float32)  # (T, D)
        else:
            raw = np.load(path, allow_pickle=True)
            if raw.dtype == object:
                data = raw.item()
                seq_flat = data["sequence"].astype(np.float32)
            else:
                seq_flat = raw.astype(np.float32)
                if seq_flat.ndim == 3:                        # (T, K, 3) legacy
                    T_, K_, D_ = seq_flat.shape
                    seq_flat = seq_flat.reshape(T_, K_ * D_)  # → (T, K*3)

        T_len, D_total = seq_flat.shape

        # Separar keypoints (primeros 126) y posición global (últimos 4 si existen)
        KPS_DIM = config.NUM_KEYPOINTS * config.KEYPOINT_DIM  # 126
        GPOS_DIM = config.GLOBAL_POS_DIM                       # 4

        if D_total >= KPS_DIM + GPOS_DIM:
            kps_flat  = seq_flat[:, :KPS_DIM]                  # (T, 126)
            global_pos = seq_flat[:, KPS_DIM:KPS_DIM + GPOS_DIM]  # (T, 4)
        elif D_total == KPS_DIM:
            # Archivo viejo sin posición global → rellenar con ceros
            kps_flat   = seq_flat
            global_pos = np.zeros((T_len, GPOS_DIM), dtype=np.float32)
        else:
            # Tamaño desconocido (ej. AUTSL legacy con pose+manos)
            # Tomar los últimos KPS_DIM valores (= las manos)
            kps_flat   = seq_flat[:, -KPS_DIM:]
            global_pos = np.zeros((T_len, GPOS_DIM), dtype=np.float32)

        # Reshape a 3D para normalización
        seq_3d = kps_flat.reshape(T_len, config.NUM_KEYPOINTS, config.KEYPOINT_DIM)

        # Normalizar SOLO los keypoints (global_pos ya está en [-0.5, 0.5])
        from utils import normalize_keypoints
        if np.max(np.abs(seq_3d[:, :, :2])) > 2.0:           # pixeles crudos
            seq_3d = normalize_keypoints(seq_3d)

        # Augmentaciones sobre los keypoints
        if self.augmentor:
            seq_3d = self.augmentor(seq_3d)

        real_len = min(T_len, self.max_seq_len)
        seq_3d   = pad_or_truncate(seq_3d, self.max_seq_len)  # (max_seq_len, 42, 3)
        T_pad    = seq_3d.shape[0]
        kps_out  = seq_3d.reshape(T_pad, KPS_DIM)             # (max_seq_len, 126)

        # Pad o truncar también global_pos
        if T_len >= self.max_seq_len:
            gpos_out = global_pos[:self.max_seq_len]
        else:
            pad_g    = np.zeros((self.max_seq_len - T_len, GPOS_DIM), dtype=np.float32)
            gpos_out = np.concatenate([global_pos, pad_g], axis=0)

        seq_flat_out = np.concatenate([kps_out, gpos_out], axis=1)  # (max_seq_len, 130)

        return {
            "sequence": torch.tensor(seq_flat_out, dtype=torch.float32),
            "label":    torch.tensor(label_idx,    dtype=torch.long),
            "length":   torch.tensor(real_len,     dtype=torch.long),
        }


# ============================================================
# Factory de DataLoaders
# ============================================================
def get_dataloaders(dataset_name: str,
                    batch_size: int = 64,
                    max_seq_len: int = 64,
                    num_workers: int = 4,
                    augment_train: bool = True,
                    max_samples: int = None,
                    class_to_idx: Dict[str, int] = None):
    """
    Crea DataLoaders para un dataset dado.

    Args:
        dataset_name: "wlasl", "autsl", o "lsp_pucp"
        batch_size: Tamaño de batch
        max_seq_len: Longitud máxima de secuencia
        num_workers: Workers para carga paralela
        augment_train: Si activar augmentaciones en train
        max_samples: Límite de muestras (para debug)
        class_to_idx: Mapeo de clases personalizado

    Returns:
        dict con "train", "val", "test" DataLoaders y "class_to_idx"
    """
    data_dir_map = {
        "wlasl": config.PROCESSED_WLASL,
        "autsl": config.PROCESSED_AUTSL,
        "lsp_pucp": config.PROCESSED_LSP,
        "lsa64": os.path.join(config.PROCESSED_DIR, "lsa64"),
    }

    if dataset_name not in data_dir_map:
        raise ValueError(f"Dataset desconocido: {dataset_name}. "
                         f"Opciones: {list(data_dir_map.keys())}")

    data_dir = data_dir_map[dataset_name]

    train_augmentor = TemporalAugmentor(enabled=augment_train)
    val_augmentor = TemporalAugmentor(enabled=False)

    # Train
    train_ds = SignSequenceDataset(
        data_dir, "train", max_seq_len, train_augmentor,
        class_to_idx=class_to_idx, max_samples=max_samples
    )

    # Si no se proporcionó class_to_idx, usar el descubierto por train
    if class_to_idx is None:
        class_to_idx = train_ds.class_to_idx

    # Val
    val_ds = SignSequenceDataset(
        data_dir, "val", max_seq_len, val_augmentor,
        class_to_idx=class_to_idx, max_samples=max_samples
    )

    # Test
    test_ds = SignSequenceDataset(
        data_dir, "test", max_seq_len, val_augmentor,
        class_to_idx=class_to_idx, max_samples=max_samples
    )

    loaders = {}
    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        if len(ds) > 0:
            loaders[name] = DataLoader(
                ds, batch_size=batch_size, shuffle=(name == "train"),
                num_workers=num_workers, pin_memory=True, drop_last=False
            )
        else:
            loaders[name] = None

    loaders["class_to_idx"] = class_to_idx
    loaders["idx_to_class"] = {v: k for k, v in class_to_idx.items()}
    loaders["num_classes"] = len(class_to_idx)

    return loaders
