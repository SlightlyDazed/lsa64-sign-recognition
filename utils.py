"""
utils.py — Funciones compartidas para el pipeline de reconocimiento de señas.
"""
import os
import cv2
import numpy as np
import torch
import logging
from pathlib import Path
from tqdm import tqdm

# ============================================================
# Logging
# ============================================================
def setup_logger(name: str, log_file: str = None, level=logging.INFO):
    """Configura un logger con salida a consola y opcionalmente a archivo."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    # Console
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # File
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


# ============================================================
# Video / Frame extraction
# ============================================================
def extract_frames(video_path: str, target_fps: int = 15,
                   frame_size: tuple = (224, 224),
                   max_frames: int = None) -> list:
    """
    Extrae frames de un video, remuestreando a target_fps y redimensionando.

    Args:
        video_path: Ruta al archivo de video
        target_fps: FPS al que muestrear
        frame_size: (width, height) de salida
        max_frames: Máximo de frames a extraer (None = sin límite)

    Returns:
        Lista de frames como arrays numpy (H, W, 3) BGR
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir el video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0:
        src_fps = 30.0  # fallback

    frame_interval = max(1, int(round(src_fps / target_fps)))
    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            frame_resized = cv2.resize(frame, frame_size)
            frames.append(frame_resized)

            if max_frames and len(frames) >= max_frames:
                break

        frame_idx += 1

    cap.release()
    return frames


# ============================================================
# Keypoint extraction — YOLOv8-pose (batched)
# ============================================================
_yolo_model = None
_yolo_model_name = None  # nombre del modelo actualmente cargado

def get_yolo_pose_model(model_name: str = "yolo11l-pose.pt"):
    """
    Carga el modelo YOLO-pose (singleton).
    Si ya hay un modelo cargado con el mismo nombre, lo reutiliza.
    Si se pide un nombre diferente, recarga. Esto evita que una llamada
    sin argumento cargue yolov8n en paralelo al yolo11l ya en GPU.
    """
    global _yolo_model, _yolo_model_name
    if _yolo_model is None or _yolo_model_name != model_name:
        from ultralytics import YOLO
        _yolo_model = YOLO(model_name)
        _yolo_model_name = model_name
    return _yolo_model


def detect_keypoints_yolo(frame: np.ndarray, model=None,
                          conf_thresh: float = 0.3) -> np.ndarray:
    """
    Detecta keypoints de persona con YOLOv8-pose en un frame.
    Returns: Array (17, 3) con (x_norm, y_norm, conf).
    """
    if model is None:
        model = get_yolo_pose_model()

    h, w = frame.shape[:2]
    results = model(frame, verbose=False, conf=conf_thresh)

    if len(results) > 0 and results[0].keypoints is not None:
        kps_data = results[0].keypoints.data
        if kps_data.shape[0] > 0:
            confs = kps_data[:, :, 2].mean(dim=1)
            best_idx = confs.argmax().item()
            kps = kps_data[best_idx].cpu().numpy()
            kps[:, 0] /= w
            kps[:, 1] /= h
            return kps

    return np.zeros((17, 3), dtype=np.float32)


def detect_keypoints_yolo_batch(frames: list, model=None,
                                conf_thresh: float = 0.3,
                                batch_size: int = 32) -> list:
    """
    Detecta keypoints de persona con YOLOv8-pose en un BATCH de frames.
    Maximiza uso de GPU enviando múltiples frames a la vez.

    Args:
        frames: Lista de imágenes BGR (H, W, 3)
        batch_size: Frames por batch de GPU (ajustar según VRAM)

    Returns:
        Lista de arrays (17, 3) por frame
    """
    if model is None:
        model = get_yolo_pose_model()

    all_kps = []

    for i in range(0, len(frames), batch_size):
        batch = frames[i:i + batch_size]
        h, w = batch[0].shape[:2]

        results = model(batch, verbose=False, conf=conf_thresh)

        for result in results:
            if result.keypoints is not None:
                kps_data = result.keypoints.data
                if kps_data.shape[0] > 0:
                    confs = kps_data[:, :, 2].mean(dim=1)
                    best_idx = confs.argmax().item()
                    kps = kps_data[best_idx].cpu().numpy()
                    kps[:, 0] /= w
                    kps[:, 1] /= h
                    all_kps.append(kps)
                else:
                    all_kps.append(np.zeros((17, 3), dtype=np.float32))
            else:
                all_kps.append(np.zeros((17, 3), dtype=np.float32))

    return all_kps


# ============================================================
# Keypoint extraction — MediaPipe Hands (Tasks API ≥0.10.14)
# ============================================================
import threading

_local_mp = threading.local()
_mp_hands_available = None  # None=not checked, True/False after check
_mp_init_lock = threading.Lock()  # serializa init entre threads — evita OOM por 8 inits simultáneas

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
HAND_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "hand_landmarker.task"
)


def _download_hand_model():
    """Descarga el modelo de manos de MediaPipe si no existe."""
    if os.path.exists(HAND_MODEL_PATH):
        return True
    try:
        import urllib.request
        os.makedirs(os.path.dirname(HAND_MODEL_PATH), exist_ok=True)
        print(f"[UTILS] Descargando hand_landmarker.task...")
        urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)
        print(f"[UTILS] ✓ Modelo descargado: {HAND_MODEL_PATH}")
        return True
    except Exception as e:
        print(f"[UTILS] ⚠ No se pudo descargar modelo de manos: {e}")
        return False


def get_mediapipe_hands():
    """
    Inicializa MediaPipe HandLandmarker con la nueva Tasks API de forma thread-safe.

    El HandLandmarker inicializa EGL/OpenGL y TFLite internamente en C++, generando
    decenas de mensajes de log (gl_context_egl, inference_feedback_manager, etc.) que
    no pueden suprimirse con variables de entorno porque ocurren antes de que Python
    tenga control. La única solución es redirigir fd=2 (stderr a nivel OS) durante
    exactamente esa llamada, y restaurarlo inmediatamente después.
    """
    global _mp_hands_available

    if _mp_hands_available is False:
        return None

    if hasattr(_local_mp, "hand_landmarker"):
        return _local_mp.hand_landmarker

    try:
        if not _download_hand_model():
            _mp_hands_available = False
            return None

        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            HandLandmarker, HandLandmarkerOptions, RunningMode
        )

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
            running_mode=RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
        )

        # Lock global: serializa la inicialización entre threads.
        # Sin esto, los 8 threads del pool inician HandLandmarker simultáneamente
        # → 8x EGL context + 8x TFLite model load en paralelo → OOM/SIGKILL.
        # Con el lock, cada thread espera su turno: RAM pico = 1 init a la vez.
        with _mp_init_lock:
            # Redirigir stderr a /dev/null durante la creación del landmarker.
            # HandLandmarker.create_from_options() escribe directo a fd=2 en C++,
            # ignorando sys.stderr y las env vars. Solo dup2 lo suprime.
            import sys
            stderr_fd = sys.stderr.fileno()
            saved_fd = os.dup(stderr_fd)
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, stderr_fd)
            os.close(devnull)
            try:
                _local_mp.hand_landmarker = HandLandmarker.create_from_options(options)
            finally:
                os.dup2(saved_fd, stderr_fd)  # siempre restaurar
                os.close(saved_fd)

        if _mp_hands_available is None:
            _mp_hands_available = True
            print("[UTILS] ✓ MediaPipe HandLandmarker inicializado (thread-safe)")

        return _local_mp.hand_landmarker
    except Exception as e:
        print(f"[UTILS] ⚠ MediaPipe Hands no disponible: {e}")
        print("[UTILS]   Usando solo YOLO-pose (17 keypoints)")
        _mp_hands_available = False
        return None


def detect_hand_keypoints(frame_rgb: np.ndarray, hands=None) -> np.ndarray:
    """
    Detecta keypoints de manos con MediaPipe Tasks API.

    Args:
        frame_rgb: Imagen RGB (H, W, 3)
        hands: Instancia de HandLandmarker opcional (si es None, se obtiene localmente)

    Returns:
        Array (42, 3): 21 keypoints mano izquierda + 21 mano derecha.
        Cada keypoint tiene (x_norm, y_norm, conf=1.0 si detectado, 0 si no).
    """
    landmarker = hands if hands is not None else get_mediapipe_hands()
    if landmarker is None:
        return np.zeros((42, 3), dtype=np.float32)

    left_kps = np.zeros((21, 3), dtype=np.float32)
    right_kps = np.zeros((21, 3), dtype=np.float32)

    try:
        import mediapipe as mp
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(frame_rgb)
        )
        result = landmarker.detect(mp_image)

        if result.hand_landmarks and result.handedness:
            for landmarks, handedness in zip(
                result.hand_landmarks, result.handedness
            ):
                label = handedness[0].category_name  # "Left" or "Right"
                img_h, img_w = frame_rgb.shape[:2]
                kps = np.array(
                    [[lm.x * img_w, lm.y * img_h, lm.z * img_w] for lm in landmarks],
                    dtype=np.float32
                )  # (21, 3)

                if label == "Left":
                    left_kps = kps
                else:
                    right_kps = kps
    except Exception:
        pass  # Silently return zeros on any error

    return np.vstack([left_kps, right_kps])  # (42, 3)


# ============================================================
# Keypoint pipeline completo por frame (legacy, single frame)
# ============================================================
def extract_keypoints_frame(frame_bgr: np.ndarray,
                            yolo_model=None,
                            use_hands: bool = True) -> np.ndarray:
    """
    Extrae keypoints de manos (sin YOLO) de un frame.
    """
    if use_hands:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return detect_hand_keypoints(frame_rgb)
    return np.zeros((0, 3), dtype=np.float32)


# ============================================================
# Secuencia de keypoints desde video — BATCHED & PARALLEL
# ============================================================
def extract_keypoints_sequence(video_path: str,
                               target_fps: int = 15,
                               frame_size: tuple = (224, 224),
                               max_frames: int = 64,
                               use_hands: bool = True,
                               yolo_model=None,
                               yolo_batch_size: int = 32) -> np.ndarray:
    """
    Extrae secuencia completa de keypoints de un video.
    Usa BATCHED YOLO (GPU) + PARALLEL MediaPipe Hands (CPU).

    Returns:
        Array (T, K, 3) donde T = número de frames, K = keypoints
    """
    frames = extract_frames(video_path, target_fps, frame_size, max_frames)

    if len(frames) == 0:
        return np.zeros((1, 42, 3), dtype=np.float32)

    if use_hands:
        # PARALLEL MediaPipe Hands — concurrent on CPU threads.
        from concurrent.futures import ThreadPoolExecutor
        def _worker(frame_bgr):
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            return detect_hand_keypoints(frame_rgb, get_mediapipe_hands())
        with ThreadPoolExecutor(max_workers=8) as pool:
            hand_kps_list = list(pool.map(_worker, frames))
        keypoints_seq = hand_kps_list
    else:
        keypoints_seq = [np.zeros((42, 3), dtype=np.float32) for _ in frames]

    return np.stack(keypoints_seq)  # (T, K, 3)


# ============================================================
# Normalización de keypoints
# ============================================================
def normalize_keypoints(kps_seq: np.ndarray) -> np.ndarray:
    """
    Normaliza una secuencia de keypoints por centroide y escala.

    IMPORTANTE: El formato ahora es (x_px, y_px, z_px) — la columna 2 es
    profundidad real (puede ser negativa), NO confianza. Se detectan puntos
    válidos comprobando si X o Y son no-cero.

    Args:
        kps_seq: (T, K, 3) — secuencia de keypoints en píxeles

    Returns:
        (T, K, 3) — normalizado por centroide y escala de X/Y
    """
    result = kps_seq.copy().astype(np.float32)

    for t in range(result.shape[0]):
        frame_kps = result[t]  # (K, 3)

        # FIX CRÍTICO: Z es profundidad real (puede ser negativo).
        # Un punto es válido si X o Y son distintos de 0.
        valid_mask = (frame_kps[:, 0] != 0) | (frame_kps[:, 1] != 0)
        if valid_mask.sum() == 0:
            continue

        valid_xy = frame_kps[valid_mask, :2]  # (N_valid, 2)
        centroid = valid_xy.mean(axis=0)       # (2,)
        scale = valid_xy.std() + 1e-6          # escalar

        # Normalizar X e Y respecto al centroide de los puntos válidos
        result[t, :, 0] = (frame_kps[:, 0] - centroid[0]) / scale
        result[t, :, 1] = (frame_kps[:, 1] - centroid[1]) / scale
        # Normalizar Z con la misma escala espacial (preserva proporciones 3D)
        result[t, :, 2] = frame_kps[:, 2] / scale
        # Los keypoints no válidos (X=Y=0) quedan normalizados a centroide/scale = negativo
        # → los ponemos en cero para que el modelo no los procese
        result[t, ~valid_mask, :] = 0.0

    return result


# ============================================================
# I/O de secuencias
# ============================================================
def save_sequence(kps_seq: np.ndarray, output_path: str):
    """Guarda secuencia de keypoints como .npy."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, kps_seq.astype(np.float32))


def load_sequence(path: str) -> np.ndarray:
    """Carga secuencia de keypoints desde .npy."""
    return np.load(path)


# ============================================================
# Padding / Truncado de secuencias
# ============================================================
def pad_or_truncate(seq: np.ndarray, target_len: int) -> np.ndarray:
    """
    Ajusta una secuencia a longitud fija.

    Args:
        seq: (T, K, 3)
        target_len: Longitud deseada

    Returns:
        (target_len, K, 3)
    """
    T = seq.shape[0]
    if T >= target_len:
        # Muestreo uniforme para mantener temporal coverage
        indices = np.linspace(0, T - 1, target_len, dtype=int)
        return seq[indices]
    else:
        # Padding con zeros al final
        pad_len = target_len - T
        padding = np.zeros((pad_len, *seq.shape[1:]), dtype=seq.dtype)
        return np.vstack([seq, padding])


# ============================================================
# Visualización
# ============================================================
def plot_confusion_matrix(y_true, y_pred, class_names, output_path: str,
                          top_n: int = None, figsize: tuple = (14, 12)):
    """Genera y guarda matriz de confusión."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    if top_n and len(class_names) > top_n:
        # Mostrar solo las top_n clases más frecuentes
        from collections import Counter
        counts = Counter(y_true)
        top_classes = [c for c, _ in counts.most_common(top_n)]
        mask = [yt in top_classes for yt in y_true]
        y_true = [yt for yt, m in zip(y_true, mask) if m]
        y_pred = [yp for yp, m in zip(y_pred, mask) if m]
        class_names = sorted(top_classes)

    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=len(class_names) <= 30, fmt="d",
                xticklabels=class_names, yticklabels=class_names,
                cmap="Blues", ax=ax)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusión")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[UTILS] Matriz de confusión guardada en {output_path}")


def plot_training_curves(train_losses, val_losses, train_accs, val_accs,
                         output_path: str):
    """Genera gráficas de loss y accuracy durante el entrenamiento."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, "b-", label="Train Loss")
    ax1.plot(epochs, val_losses, "r-", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss por Epoch")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_accs, "b-", label="Train Acc")
    ax2.plot(epochs, val_accs, "r-", label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy por Epoch")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[UTILS] Curvas de entrenamiento guardadas en {output_path}")


# ============================================================
# Checkpointing
# ============================================================
def save_checkpoint(model, optimizer, epoch, metrics, path):
    """Guarda checkpoint del modelo."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }, path)
    print(f"[UTILS] Checkpoint guardado: {path}")


def load_checkpoint(model, path, optimizer=None, device="cpu"):
    """Carga checkpoint del modelo."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if optimizer and "optimizer_state_dict" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except ValueError:
            print("[UTILS] No se pudo cargar el estado del optimizer (cambio de arquitectura)")
    return ckpt.get("epoch", 0), ckpt.get("metrics", {})
