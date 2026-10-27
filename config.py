"""
config.py — Configuración centralizada del pipeline de reconocimiento de lengua de señas.

Cambios v2:
  - MODEL_HIDDEN_DIM: 256 → 512  (mayor capacidad expresiva)
  - MODEL_NUM_LSTM_LAYERS: 2 → 3
  - Batch sizes aumentados para aprovechar VRAM disponible
  - train_config de LSA64/AUTSL/WLASL incluye parámetros de Mixup y WarmRestarts
"""
import os
import torch

# ============================================================
# Rutas base
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Archivos fuente (zips/tars)
SOURCE_FILES = {
    "wlasl_zip":         os.path.join(PROJECT_ROOT, "WLASL-master.zip"),
    "autsl_zip":         os.path.join(PROJECT_ROOT, "archive.zip"),
    "lsp_videos_tar":    os.path.join(PROJECT_ROOT, "XXXX_LSP_peru1235_Videos.tar"),
    "lsp_keypoints_tar": os.path.join(PROJECT_ROOT, "XXXX_LSP_peru1235_Keypoints.tar"),
    "lsp_srt_tar":       os.path.join(PROJECT_ROOT, "XXXX_LSP_peru1235_SRT.tar"),
}

# Directorios de datos
DATA_DIR      = os.path.join(PROJECT_ROOT, "data")
RAW_DIR       = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

RAW_WLASL = os.path.join(RAW_DIR, "wlasl")
RAW_AUTSL = os.path.join(RAW_DIR, "autsl")
RAW_LSP   = os.path.join(RAW_DIR, "lsp_pucp")

PROCESSED_WLASL = os.path.join(PROCESSED_DIR, "wlasl")
PROCESSED_AUTSL = os.path.join(PROCESSED_DIR, "autsl")
PROCESSED_LSP   = os.path.join(PROCESSED_DIR, "lsp_pucp")

# Modelos y resultados
MODELS_DIR  = os.path.join(PROJECT_ROOT, "models")
EXPORT_DIR  = os.path.join(MODELS_DIR, "export")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# ============================================================
# Device
# ============================================================
def get_device():
    """Detecta el mejor dispositivo disponible (ROCm/CUDA/CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name   = torch.cuda.get_device_name(0)
        print(f"[CONFIG] Usando GPU: {name}")
    else:
        device = torch.device("cpu")
        print("[CONFIG] GPU no disponible, usando CPU")
    return device

DEVICE = get_device()

# ============================================================
# Preprocesamiento
# ============================================================
FRAME_SIZE   = (224, 224)
TARGET_FPS   = 15
MAX_SEQ_LEN  = 64
MIN_SEQ_LEN  = 8

# Keypoints: Usaremos EXCLUSIVAMENTE MediaPipe Hands (copia exacta de monitor_emg)
NUM_YOLO_KEYPOINTS     = 0
KEYPOINT_DIM           = 3    # (x_px, y_px, z_px)

USE_MEDIAPIPE_HANDS    = True
NUM_MEDIAPIPE_HAND_KPS = 21

def get_num_keypoints():
    if USE_MEDIAPIPE_HANDS:
        return NUM_YOLO_KEYPOINTS + 2 * NUM_MEDIAPIPE_HAND_KPS  # 42
    return NUM_YOLO_KEYPOINTS

NUM_KEYPOINTS = get_num_keypoints()
# 4 valores extra: posición relativa de muñeca izq+der respecto al centro del frame
# Permite que el modelo sepa si la mano está cerca de la cabeza, pecho o cintura
GLOBAL_POS_DIM = 4   # [lw_x, lw_y, rw_x, rw_y] en rango [-0.5, 0.5]
INPUT_DIM = NUM_KEYPOINTS * KEYPOINT_DIM + GLOBAL_POS_DIM  # 126 + 4 = 130

# ============================================================
# WLASL
# ============================================================
WLASL_JSON        = "WLASL_v0.3.json"
WLASL_NSLT_JSON   = "nslt_100.json"
WLASL_CLASS_LIST  = "wlasl_class_list.txt"
WLASL_NUM_CLASSES = 100

# ============================================================
# AUTSL
# ============================================================
AUTSL_NUM_CLASSES = 226
AUTSL_TRAIN_CSV   = "train.csv"
AUTSL_VAL_CSV     = "val.csv"
AUTSL_TEST_CSV    = "test.csv"

# ============================================================
# LSP PUCP
# ============================================================
LSP_MIN_SAMPLES_PER_CLASS = 3
LSP_TRAIN_RATIO = 0.7
LSP_VAL_RATIO   = 0.15
LSP_TEST_RATIO  = 0.15

# ============================================================
# Modelo — Clasificador temporal  ← CAMBIOS PRINCIPALES v2
# ============================================================
MODEL_HIDDEN_DIM       = 512   # ↑ era 256 — mayor capacidad con buena VRAM
MODEL_NUM_LSTM_LAYERS  = 3     # ↑ era 2
MODEL_DROPOUT          = 0.3
MODEL_BIDIRECTIONAL    = True

# ============================================================
# Parámetros comunes de mejora (Mixup + WarmRestarts)
# Estos valores se inyectan en train_config de cada etapa.
# ============================================================
TRAIN_IMPROVEMENTS = {
    "use_mixup":           True,
    "mixup_alpha":         0.3,
    "focal_gamma":         2.0,
    "warmrestart_T0":      10,
    "warmrestart_Tmult":   2,
    "early_stop_patience": 15,
}

# ============================================================
# Entrenamiento
# ============================================================

# Etapa 1: WLASL pre-training
WLASL_TRAIN = {
    "epochs":      50,
    "batch_size":  128,   # ↑ era 64
    "lr":          1e-3,
    "weight_decay": 1e-4,
    "scheduler":   "cosine_warmrestart",
    "checkpoint":  os.path.join(MODELS_DIR, "wlasl_base.pth"),
    **TRAIN_IMPROVEMENTS,
}

# Etapa 2: AUTSL fine-tuning
AUTSL_TRAIN = {
    "epochs":        30,
    "batch_size":    128,   # ↑ era 64
    "lr":            1e-4,
    "weight_decay":  1e-4,
    "freeze_epochs": 10,
    "scheduler":     "cosine_warmrestart",
    "checkpoint":    os.path.join(MODELS_DIR, "autsl_finetuned.pth"),
    **TRAIN_IMPROVEMENTS,
}

# Etapa 3: LSP fine-tuning
LSP_TRAIN = {
    "epochs":        40,
    "batch_size":    64,    # LSP es más pequeño, batch moderado
    "lr":            5e-5,
    "weight_decay":  1e-4,
    "freeze_epochs": 5,
    "scheduler":     "cosine_warmrestart",
    "checkpoint":    os.path.join(MODELS_DIR, "lsp_final.pth"),
    **TRAIN_IMPROVEMENTS,
}

# ============================================================
# YOLO pose model
# ============================================================
YOLO_POSE_MODEL = "yolo11l-pose.pt"

# ============================================================
# Crear directorios necesarios
# ============================================================
for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, RAW_WLASL, RAW_AUTSL, RAW_LSP,
          PROCESSED_WLASL, PROCESSED_AUTSL, PROCESSED_LSP,
          MODELS_DIR, EXPORT_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)
