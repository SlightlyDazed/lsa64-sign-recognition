"""
02_preprocess_frames.py — Extrae keypoints con YOLOv8/11-pose (GPU) + MediaPipe Hands (CPU).

Arquitectura optimizada:
  - Videos se procesan UNO A LA VEZ para evitar contención de GPU
  - TODOS los frames de un video se pasan a YOLO en UNA sola llamada (batch interno)
  - MediaPipe Hands corre en PARALELO en CPU (ThreadPool, instancia thread-local reutilizada)
  - Pipeline: [CPU: decode video] → [GPU: YOLO batch] → [CPU‖: MediaPipe parallel] → save
  - Modelo YOLO se carga UNA SOLA VEZ en main() y se pasa a cada función

Uso:
    python 02_preprocess_frames.py --dataset autsl
    python 02_preprocess_frames.py --dataset all --max-videos 100
    python 02_preprocess_frames.py --dataset autsl --no-hands  # 2x más rápido
"""
import os
import sys

# Silenciar logs verbosos de MediaPipe/TFLite/glog ANTES de importarlos.
# Sin esto aparecen decenas de "inference_feedback_manager" y "EGL initialized" por video.
os.environ["GLOG_minloglevel"]    = "3"   # suprime INFO/WARNING/ERROR de glog (C++)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suprime TFLite warnings
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1" # fuerza CPU en MediaPipe (evita competencia con YOLO en GPU)

import json
import csv
import argparse
import traceback
import threading
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from utils import (
    setup_logger, normalize_keypoints, save_sequence,
    get_yolo_pose_model, get_mediapipe_hands, detect_hand_keypoints,
)

# Suprimir warnings de absl (usado internamente por MediaPipe)
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except ImportError:
    pass

logger = setup_logger("preprocess", os.path.join(config.RESULTS_DIR, "02_preprocess.log"))


# ============================================================
# Warm-up centralizado (llamar UNA sola vez desde main)
# ============================================================
def warmup_yolo(yolo_model):
    """
    Dispara la compilación de kernels HIP/ROCm con ruido realista.
    Usar np.zeros puede no compilar todos los kernels en ROCm.
    Se pasa como lista para respetar la misma firma que en inferencia real.
    """
    dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    _ = yolo_model([dummy], verbose=False)
    logger.info("  ✓ YOLO warm-up completo (ROCm kernels compilados)")


# ============================================================
# Worker thread-local para MediaPipe
# ============================================================
# Una instancia de Hands por thread, reutilizada en todos los frames de ese thread.
# Crear una instancia por frame inicializaba TFLite repetidamente → lento.
_thread_local = threading.local()

def _get_thread_hands():
    """Devuelve la instancia de MediaPipe Hands de este thread, creándola si no existe."""
    if not hasattr(_thread_local, "hands"):
        _thread_local.hands = get_mediapipe_hands()
    return _thread_local.hands

def _mediapipe_worker(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Worker para ThreadPoolExecutor.
    - Convierte BGR→RGB aquí (evita copia masiva previa de todos los frames)
    - Usa instancia thread-local reutilizable (1 instancia por thread, no por frame)
    - MediaPipe NO es thread-safe si se comparte una instancia entre threads distintos.

    IMPORTANTE: detect_hand_keypoints(frame_rgb, hands_instance) debe aceptar dos args.
    Si en utils.py la firma es detect_hand_keypoints(frame_rgb), actualiza utils.py:
      def detect_hand_keypoints(frame_rgb, hands=None)
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    hands_instance = _get_thread_hands()
    return detect_hand_keypoints(frame_rgb, hands_instance)


# ============================================================
# Procesador principal: batch YOLO + parallel MediaPipe
# ============================================================
def process_single_video(video_path: str, output_path: str,
                         yolo_model, use_hands: bool,
                         mp_workers: int = 8,
                         pool: ThreadPoolExecutor = None) -> bool:
    """
    Procesa un video:
      1. Decode frames (CPU) — grab()+retrieve() para saltar frames sin decodificar
      2. YOLO pose en TODOS los frames a la vez (GPU, una sola llamada)
      3. MediaPipe Hands en paralelo (CPU, pool externo persistente o local de fallback)
      4. Normalizar + guardar .npy (crea directorios si no existen)

    pool: ThreadPoolExecutor externo persistente (recomendado). Si es None se crea
          uno local que se destruye al terminar — los threads mueren y MediaPipe
          reinicializa EGL completo en el siguiente video (lento).
    """
    try:
        # ----------------------------------------------------------
        # 1) Decode video — grab()+retrieve() es la forma correcta.
        #    cap.read() = grab()+retrieve() internamente. Si además llamas
        #    grab() extra para saltar frames, desincronizas frame_idx con la
        #    posición real del video. Usando siempre grab() para avanzar y
        #    retrieve() solo cuando necesitas el frame, el conteo es exacto.
        # ----------------------------------------------------------
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(f"  No se pudo abrir: {Path(video_path).name}")
            return False

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = max(1, int(round(src_fps / config.TARGET_FPS)))
        frames = []
        frame_idx = 0

        while True:
            grabbed = cap.grab()
            if not grabbed:
                break
            if frame_idx % frame_interval == 0:
                ret, frame = cap.retrieve()
                if ret:
                    frames.append(frame)
                    if len(frames) >= config.MAX_SEQ_LEN:
                        break
            frame_idx += 1

        cap.release()

        if len(frames) == 0:
            logger.warning(f"  Sin frames: {Path(video_path).name}")
            return False

        # ----------------------------------------------------------
        # 2) YOLO pose — omitir completamente si no se usan
        # ----------------------------------------------------------
        pose_kps_list = []
        if config.NUM_YOLO_KEYPOINTS > 0:
            YOLO_BATCH = 16
            all_results = []
            for i in range(0, len(frames), YOLO_BATCH):
                batch = frames[i:i+YOLO_BATCH]
                all_results.extend(yolo_model(batch, verbose=False, conf=0.3))
            
            for i, result in enumerate(all_results):
                h, w = frames[i].shape[:2]
                if result.keypoints is not None and result.keypoints.data.shape[0] > 0:
                    kps_data = result.keypoints.data
                    confs = kps_data[:, :, 2].mean(dim=1)
                    best_idx = confs.argmax().item()
                    kps = kps_data[best_idx].cpu().numpy()
                    kps[:, 0] /= w
                    kps[:, 1] /= h
                    pose_kps_list.append(kps)
                else:
                    pose_kps_list.append(np.zeros((17, 3), dtype=np.float32))
        else:
            pose_kps_list = [np.zeros((0, 3), dtype=np.float32) for _ in frames]

        # ----------------------------------------------------------
        # 3) MediaPipe Hands — parallel CPU, instancia thread-local reutilizada.
        #    Si se pasa un pool externo (persistente entre videos), los threads
        #    sobreviven entre llamadas y _get_thread_hands() no reinicializa EGL.
        #    Sin pool externo, se crea uno local (comportamiento anterior: lento).
        # ----------------------------------------------------------
        if use_hands:
            if pool is not None:
                hand_kps_list = list(pool.map(_mediapipe_worker, frames))
            else:
                with ThreadPoolExecutor(max_workers=mp_workers) as _local_pool:
                    hand_kps_list = list(_local_pool.map(_mediapipe_worker, frames))
            keypoints_seq = [
                np.vstack([p, h]) for p, h in zip(pose_kps_list, hand_kps_list)
            ]
        else:
            keypoints_seq = pose_kps_list

        # ----------------------------------------------------------
        # 4) Normalizar + guardar
        #    makedirs: evita fallo silencioso cuando el dir no existe
        # ----------------------------------------------------------
        seq = np.stack(keypoints_seq)   # (T, K, 3)
        seq = normalize_keypoints(seq)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_sequence(seq, output_path)
        return True

    except Exception as e:
        logger.warning(
            f"  Error en {Path(video_path).name}: {e}\n"
            f"{traceback.format_exc()}"
        )
        return False


# ============================================================
# AUTSL
# ============================================================
def preprocess_autsl(max_videos=None, use_hands=True, yolo_model=None):
    logger.info("=" * 50)
    logger.info("Preprocesando AUTSL")
    logger.info("=" * 50)

    autsl_root = config.RAW_AUTSL
    csv_files = list(Path(autsl_root).rglob("train.csv"))
    if not csv_files:
        logger.error("train.csv no encontrado")
        return
    base_dir = csv_files[0].parent
    logger.info(f"  Base: {base_dir}")

    if yolo_model is None:
        yolo_model = get_yolo_pose_model(config.YOLO_POSE_MODEL)

    for split_name, csv_name in [("train", "train.csv"),
                                  ("val",   "val.csv"),
                                  ("test",  "test.csv")]:
        csv_path = base_dir / csv_name
        if not csv_path.exists():
            continue

        with open(csv_path) as f:
            rows = list(csv.reader(f))
        logger.info(f"  {split_name}: {len(rows)} entradas en CSV")

        video_dirs = [base_dir / split_name, base_dir]

        pending = []
        for row in rows:
            if max_videos and len(pending) >= max_videos:
                break
            if len(row) < 2:
                continue
            filename, label_id = row[0].strip(), row[1].strip()
            class_name = f"class_{int(label_id):03d}"

            video_path = None
            for vdir in video_dirs:
                for ext in ["", ".mp4"]:
                    p = vdir / (filename + ext)
                    if p.exists():
                        video_path = str(p)
                        break
                if video_path:
                    break
            if not video_path:
                continue

            out_path = os.path.join(
                config.PROCESSED_AUTSL, "sequences", split_name, class_name,
                f"{Path(filename).stem}.npy"
            )
            if os.path.exists(out_path):
                continue
            pending.append((video_path, out_path))

        logger.info(f"  {split_name}: {len(pending)} pendientes")
        if not pending:
            continue

        ok, fail = 0, 0
        # Pool persistente: los threads viven durante todo el split.
        # _get_thread_hands() crea la instancia MediaPipe UNA vez por thread
        # y la reutiliza en todos los videos — sin reinit EGL entre videos.
        mp_workers = 8
        with ThreadPoolExecutor(max_workers=mp_workers) as mp_pool:
            pbar = tqdm(pending, desc=f"AUTSL-{split_name}")
            for video_path, out_path in pbar:
                if process_single_video(video_path, out_path, yolo_model,
                                        use_hands, mp_workers, pool=mp_pool):
                    ok += 1
                else:
                    fail += 1
                pbar.set_postfix(ok=ok, fail=fail)

        logger.info(f"  AUTSL {split_name}: {ok} OK, {fail} fallos")


# ============================================================
# LSP PUCP
# ============================================================
def preprocess_lsp(max_videos=None, use_hands=True, yolo_model=None):
    logger.info("=" * 50)
    logger.info("Preprocesando LSP PUCP")
    logger.info("=" * 50)

    lsp_root = config.RAW_LSP
    seg_dirs = list(Path(lsp_root).rglob("SEGMENTED_SIGN_ADJUSTED"))
    if not seg_dirs:
        logger.error("SEGMENTED_SIGN_ADJUSTED no encontrado")
        return
    seg_dir = seg_dirs[0]

    all_videos = []
    for story_dir in sorted(seg_dir.iterdir()):
        if not story_dir.is_dir():
            continue
        for vf in sorted(story_dir.glob("*.mp4")):
            stem = vf.stem
            parts = stem.rsplit("_", 1)
            label = parts[0] if (len(parts) == 2 and parts[1].isdigit()) else stem
            label = label.strip().upper()
            if label:
                all_videos.append((str(vf), label))

    label_counts = Counter(l for _, l in all_videos)
    valid_labels = {l for l, c in label_counts.items()
                    if c >= config.LSP_MIN_SAMPLES_PER_CLASS}
    filtered = [(p, l) for p, l in all_videos if l in valid_labels]
    logger.info(f"  Clases: {len(valid_labels)}, videos: {len(filtered)}")

    videos_by_class = defaultdict(list)
    for path, label in filtered:
        videos_by_class[label].append(path)

    np.random.seed(42)
    splits = {"train": [], "val": [], "test": []}
    for label, paths in sorted(videos_by_class.items()):
        np.random.shuffle(paths)
        n = len(paths)
        n_train = max(1, int(n * config.LSP_TRAIN_RATIO))
        n_val   = max(1, int(n * config.LSP_VAL_RATIO))
        t  = paths[:n_train]
        v  = paths[n_train:n_train + n_val]
        te = paths[n_train + n_val:]
        # FIX: no mutar v con pop(), usar slicing limpio
        if not te and len(v) > 1:
            te = [v[-1]]
            v  = v[:-1]
        elif not te:
            te = [t[-1]]
        for p in t:  splits["train"].append((p, label))
        for p in v:  splits["val"].append((p, label))
        for p in te: splits["test"].append((p, label))

    for s, items in splits.items():
        logger.info(f"  {s}: {len(items)} videos")

    sorted_classes = sorted(valid_labels)
    class_to_idx = {c: i for i, c in enumerate(sorted_classes)}
    map_path = os.path.join(config.PROCESSED_LSP, "class_mapping.json")
    os.makedirs(os.path.dirname(map_path), exist_ok=True)
    with open(map_path, "w") as f:
        json.dump(class_to_idx, f, ensure_ascii=False, indent=2)

    if yolo_model is None:
        yolo_model = get_yolo_pose_model(config.YOLO_POSE_MODEL)

    for split_name, items in splits.items():
        pending = []
        for vp, label in items:
            if max_videos and len(pending) >= max_videos:
                break
            vid_id = Path(vp).stem
            out = os.path.join(config.PROCESSED_LSP, "sequences", split_name,
                               label, f"{vid_id}.npy")
            if not os.path.exists(out):
                pending.append((vp, out))

        logger.info(f"  {split_name}: {len(pending)} pendientes")
        if not pending:
            continue

        ok = 0
        mp_workers = 8
        with ThreadPoolExecutor(max_workers=mp_workers) as mp_pool:
            pbar = tqdm(pending, desc=f"LSP-{split_name}")
            for vp, out in pbar:
                if process_single_video(vp, out, yolo_model, use_hands,
                                        mp_workers, pool=mp_pool):
                    ok += 1
                pbar.set_postfix(ok=ok)

        logger.info(f"  LSP {split_name}: {ok} procesados")


# ============================================================
# WLASL
# ============================================================
def preprocess_wlasl(max_videos=None, use_hands=True, yolo_model=None):
    logger.info("=" * 50)
    logger.info("Preprocesando WLASL")
    logger.info("=" * 50)

    wlasl_root = config.RAW_WLASL
    nslt_path = next(Path(wlasl_root).rglob(config.WLASL_NSLT_JSON), None)
    if not nslt_path:
        logger.error(f"{config.WLASL_NSLT_JSON} no encontrado")
        return

    cls_path = next(Path(wlasl_root).rglob(config.WLASL_CLASS_LIST), None)
    idx_to_class = {}
    if cls_path:
        with open(cls_path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    idx_to_class[int(parts[0])] = parts[1]

    with open(nslt_path) as f:
        nslt_data = json.load(f)

    json_path = next(Path(wlasl_root).rglob(config.WLASL_JSON), None)
    video_paths = {}
    if json_path:
        with open(json_path) as f:
            wlasl_full = json.load(f)
        existing = {p.stem: str(p) for p in Path(wlasl_root).rglob("*.mp4")}
        for entry in wlasl_full:
            for inst in entry.get("instances", []):
                vid = str(inst.get("video_id", ""))
                if vid in existing:
                    video_paths[vid] = existing[vid]

    if not video_paths:
        logger.warning("  WLASL sin videos locales, saltando...")
        return

    if yolo_model is None:
        yolo_model = get_yolo_pose_model(config.YOLO_POSE_MODEL)

    pending = []
    for vid_id, entry in nslt_data.items():
        if max_videos and len(pending) >= max_videos:
            break
        if vid_id not in video_paths:
            continue
        cls_id   = entry["action"][0]
        split    = entry.get("subset", "train")
        cls_name = idx_to_class.get(cls_id, f"class_{cls_id}")
        out = os.path.join(config.PROCESSED_WLASL, "sequences", split,
                           cls_name, f"{vid_id}.npy")
        if not os.path.exists(out):
            pending.append((video_paths[vid_id], out))

    logger.info(f"  Pendientes: {len(pending)}")
    if pending:
        ok = 0
        mp_workers = 8
        with ThreadPoolExecutor(max_workers=mp_workers) as mp_pool:
            for vp, out in tqdm(pending, desc="WLASL"):
                if process_single_video(vp, out, yolo_model, use_hands,
                                        mp_workers, pool=mp_pool):
                    ok += 1
        logger.info(f"  WLASL: {ok} procesados")


# ============================================================
# Main — carga YOLO y hace warm-up UNA SOLA VEZ
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Preprocesa videos con YOLO-pose (GPU) + MediaPipe (CPU)")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["wlasl", "autsl", "lsp_pucp", "all"])
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--no-hands", action="store_true",
                        help="Solo YOLO body keypoints, sin MediaPipe Hands (2x más rápido)")
    args = parser.parse_args()

    use_hands = not args.no_hands
    logger.info(f"  Modelo: {config.YOLO_POSE_MODEL}")
    logger.info(f"  Hands: {use_hands}")
    logger.info(f"  Max frames/video: {config.MAX_SEQ_LEN}")

    # Carga y warm-up UNA SOLA VEZ para todos los datasets
    yolo_model = get_yolo_pose_model(config.YOLO_POSE_MODEL)
    warmup_yolo(yolo_model)

    if args.dataset in ("wlasl", "all"):
        preprocess_wlasl(args.max_videos, use_hands, yolo_model)
    if args.dataset in ("autsl", "all"):
        preprocess_autsl(args.max_videos, use_hands, yolo_model)
    if args.dataset in ("lsp_pucp", "all"):
        preprocess_lsp(args.max_videos, use_hands, yolo_model)

    logger.info("✓ Preprocesamiento completado.")


if __name__ == "__main__":
    main()
