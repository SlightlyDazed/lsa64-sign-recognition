"""
extract_lsa64.py — Extracción de keypoints MediaPipe del dataset LSA64.
VERSION: multiprocessing v2 — fixes timestamp + EGL contention.

Fixes aplicados:
  1. RunningMode.IMAGE en vez de VIDEO → elimina el error
     "Input timestamp must be monotonically increasing" al compartir
     el landmarker entre videos en el mismo worker.
  2. Sin inicialización EGL/OpenGL por worker → evita el crash
     "amdgpu: failed to create IB buffer" cuando 5 workers compiten
     por el mismo contexto gráfico de la RX 9070.
  3. Modelo instanciado UNA vez por worker (no por video).
  4. Pool de procesos paralelos — usa todos los núcleos disponibles.
  5. Barra de progreso con tqdm + reporte de velocidad (videos/s).
  6. Reanudación automática (skip si .npy ya existe).
  7. --num-workers para controlar paralelismo explícitamente.

Nota sobre RunningMode.IMAGE vs VIDEO:
  VIDEO aplica un filtro temporal entre frames (suavizado de landmarks)
  que requiere timestamps estrictamente crecientes y continua entre videos.
  IMAGE trata cada frame de forma independiente — correcto para nuestro
  caso donde cada video es una muestra separada, y compatible con
  multiprocessing sin estado compartido entre videos.

Uso:
    python extract_lsa64.py [--input-dir /ruta/a/all] [--output-dir processed/lsa64]
    python extract_lsa64.py --num-workers 6   # default: cpu_count() - 1
    python extract_lsa64.py --dry-run          # solo estadísticas
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count, current_process
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# ─── Constantes del dataset ────────────────────────────────────────────────────

LSA64_CLASS_NAMES = {
    1:  "opaque",      2:  "red",          3:  "green",       4:  "yellow",
    5:  "bright",      6:  "light-blue",   7:  "colors",      8:  "pink",
    9:  "women",       10: "enemy",        11: "son",         12: "man",
    13: "away",        14: "drawer",       15: "born",        16: "learn",
    17: "call",        18: "skimmer",      19: "bitter",      20: "sweet-milk",
    21: "milk",        22: "water",        23: "food",        24: "argentina",
    25: "uruguay",     26: "country",      27: "last-name",   28: "where",
    29: "mock",        30: "birthday",     31: "breakfast",   32: "photo",
    33: "hungry",      34: "map",          35: "coin",        36: "music",
    37: "ship",        38: "none",         39: "name",        40: "patience",
    41: "perfume",     42: "deaf",         43: "trap",        44: "rice",
    45: "barbecue",    46: "candy",        47: "chewing-gum", 48: "spaghetti",
    49: "yogurt",      50: "accept",       51: "thanks",      52: "shut-down",
    53: "appear",      54: "to-land",      55: "catch",       56: "help",
    57: "dance",       58: "bathe",        59: "buy",         60: "copy",
    61: "run",         62: "realize",      63: "give",        64: "find",
}

TRAIN_SUBJECTS = {1, 2, 3, 4, 5, 6}
VAL_SUBJECTS   = {7, 8}
TEST_SUBJECTS  = {9, 10}

HOLISTIC_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "holistic_landmarker.task"
)
HOLISTIC_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "holistic_landmarker/holistic_landmarker/float16/latest/"
    "holistic_landmarker.task"
)

# ─── Estado global POR WORKER (inicializado una vez en el proceso hijo) ────────
# Cada proceso del pool tiene su propia copia — sin locks necesarios.
_worker_landmarker = None
_worker_mp         = None


def _worker_init(model_path: str):
    """
    Inicializador del pool: carga MediaPipe UNA SOLA VEZ por proceso hijo.

    Usa RunningMode.IMAGE (no VIDEO):
      - IMAGE no requiere timestamps → elimina "Input timestamp must be
        monotonically increasing" al procesar múltiples videos.
      - IMAGE no inicializa EGL/OpenGL → evita "amdgpu: failed to create
        IB buffer" cuando varios workers compiten por la RX 9070.
    """
    global _worker_landmarker, _worker_mp

    # Bloquear EGL/OpenGL ANTES de importar mediapipe o cualquier lib gráfica.
    # MediaPipe siempre intenta crear contexto GPU al cargar holistic_landmarker,
    # independientemente del delegate. Con DISPLAY="" y EGL_PLATFORM=surfaceless
    # forzamos un contexto offscreen mínimo que no compite con otros workers.
    os.environ["GLOG_minloglevel"]      = "3"
    os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
    os.environ["DISPLAY"]               = ""       # sin display X11
    os.environ["EGL_PLATFORM"]          = "surfaceless"  # EGL offscreen, sin GPU real
    os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"      # forzar Mesa software renderer
    os.environ["GALLIUM_DRIVER"]        = "llvmpipe"     # llvmpipe (CPU puro)

    import mediapipe as mp
    _worker_mp = mp

    VisionRunningMode  = mp.tasks.vision.RunningMode
    HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
    HolisticOptions    = mp.tasks.vision.HolisticLandmarkerOptions
    BaseOptions        = mp.tasks.BaseOptions

    options = HolisticOptions(
        base_options=BaseOptions(
            model_asset_path=model_path,
            delegate=BaseOptions.Delegate.CPU,  # ← forzar CPU, evita EGL/amdgpu crash
        ),
        running_mode=VisionRunningMode.IMAGE,   # ← IMAGE no VIDEO (sin timestamps)
        min_face_detection_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
    )
    _worker_landmarker = HolisticLandmarker.create_from_options(options)

    pid = current_process().pid
    print(f"  [Worker PID {pid}] MediaPipe listo ✓ (IMAGE/CPU)", flush=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def parse_filename(filename):
    """CCC_SSS_RRR.mp4 → (class_id, subject_id, rep_id) o None."""
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) != 3:
        return None
    try:
        return int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None


def get_split(subject_id):
    if subject_id in TRAIN_SUBJECTS:
        return "train"
    if subject_id in VAL_SUBJECTS:
        return "val"
    if subject_id in TEST_SUBJECTS:
        return "test"
    return None


def ensure_holistic_model():
    if os.path.exists(HOLISTIC_MODEL_PATH):
        return
    print(f"Descargando modelo MediaPipe Holistic...")
    print(f"  URL: {HOLISTIC_MODEL_URL}")
    import urllib.request
    urllib.request.urlretrieve(HOLISTIC_MODEL_URL, HOLISTIC_MODEL_PATH)
    print(f"  Guardado en: {HOLISTIC_MODEL_PATH}")


# ─── Extracción de keypoints (usa el landmarker ya cargado en el worker) ──────

def _extract_frames(video_path: Path, max_seq_len: int) -> np.ndarray | None:
    """
    Extrae keypoints frame a frame usando el landmarker global del worker.
    Retorna array (T, INPUT_DIM) o None si el video está vacío/corrupto.
    """
    import cv2
    mp = _worker_mp

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    keypoints_seq = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = _worker_landmarker.detect(mp_image)
        frame_idx += 1

        img_h, img_w = frame_rgb.shape[:2]

        # --- Posición global de la muñeca (índice 0 de cada mano) ---
        # Guardamos ANTES de normalizar → en espacio normalizado [0,1] de MediaPipe
        # Centrado en 0 restando 0.5, para que el modelo vea posición relativa al frame
        if results.left_hand_landmarks:
            lw = results.left_hand_landmarks[0]   # muñeca = landmark 0
            lw_pos = np.array([lw.x - 0.5, lw.y - 0.5], dtype=np.float32)
        else:
            lw_pos = np.zeros(2, dtype=np.float32)

        if results.right_hand_landmarks:
            rw = results.right_hand_landmarks[0]
            rw_pos = np.array([rw.x - 0.5, rw.y - 0.5], dtype=np.float32)
        else:
            rw_pos = np.zeros(2, dtype=np.float32)

        global_pos = np.concatenate([lw_pos, rw_pos])  # (4,)

        # --- Keypoints 3D en píxeles ---
        # Mano izquierda: 21 × 3 = 63
        if results.left_hand_landmarks:
            lh = np.array(
                [[lm.x * img_w, lm.y * img_h, lm.z * img_w] for lm in results.left_hand_landmarks],
                dtype=np.float32
            ).flatten()
        else:
            lh = np.zeros(63, dtype=np.float32)

        # Mano derecha: 21 × 3 = 63
        if results.right_hand_landmarks:
            rh = np.array(
                [[lm.x * img_w, lm.y * img_h, lm.z * img_w] for lm in results.right_hand_landmarks],
                dtype=np.float32
            ).flatten()
        else:
            rh = np.zeros(63, dtype=np.float32)

        # 126 kps manos + 4 posición global = 130 dims (= config.INPUT_DIM)
        frame_kp = np.concatenate([lh, rh, global_pos])
        dim = config.INPUT_DIM
        if len(frame_kp) > dim:
            frame_kp = frame_kp[:dim]
        elif len(frame_kp) < dim:
            frame_kp = np.pad(frame_kp, (0, dim - len(frame_kp)))

        keypoints_seq.append(frame_kp)
        if len(keypoints_seq) >= max_seq_len:
            break

    cap.release()
    return np.array(keypoints_seq, dtype=np.float32) if keypoints_seq else None


# ─── Función ejecutada por cada worker del pool ───────────────────────────────

def _process_sample(task: tuple) -> dict:
    """
    Procesa un video y guarda el .npy.
    Retorna dict con resultado para el proceso principal.
    """
    vpath, class_id, subject_id, rep_id, split, output_dir, max_seq_len, \
        class_to_idx = task

    class_name = LSA64_CLASS_NAMES[class_id]
    class_idx  = class_to_idx[class_name]
    out_name   = f"{class_idx:03d}_s{subject_id:02d}_r{rep_id:02d}.npy"
    out_path   = Path(output_dir) / "sequences" / split / class_name / out_name

    # Skip si ya existe (reanudación automática)
    if out_path.exists():
        return {"status": "skip", "name": out_name, "split": split}

    try:
        seq = _extract_frames(vpath, max_seq_len)
    except Exception as e:
        return {"status": "error", "name": out_name, "split": split,
                "error": str(e)}

    if seq is None or len(seq) == 0:
        return {"status": "empty", "name": out_name, "split": split}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), {
        "sequence":   seq,
        "label":      class_idx,
        "length":     len(seq),
        "class_name": class_name,
    })

    return {
        "status": "ok",
        "name":   out_name,
        "split":  split,
        "frames": len(seq),
        "class":  class_name,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extrae keypoints de LSA64 con multiprocessing"
    )
    parser.add_argument("--input-dir", type=str,
                        default=os.path.join(
                            os.path.dirname(os.path.abspath(__file__)), "all"),
                        help="Carpeta con los videos .mp4 de LSA64")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(config.PROCESSED_DIR, "lsa64"),
                        help="Carpeta de salida para .npy y JSONs")
    parser.add_argument("--max-seq-len", type=int, default=config.MAX_SEQ_LEN)
    parser.add_argument("--num-workers", type=int,
                        default=max(1, cpu_count() - 1),
                        help="Número de procesos paralelos (default: núcleos - 1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Solo muestra estadísticas, sin procesar")
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"ERROR: No se encontró la carpeta de videos: {input_dir}")
        sys.exit(1)

    # ── Escanear y clasificar videos ──────────────────────────────────────────
    all_videos  = sorted(input_dir.glob("*.mp4"))
    print(f"Videos encontrados: {len(all_videos)}")

    split_counts = {"train": 0, "val": 0, "test": 0, "skip": 0}
    samples = []

    for vpath in all_videos:
        parsed = parse_filename(vpath.name)
        if parsed is None:
            split_counts["skip"] += 1
            continue
        class_id, subject_id, rep_id = parsed
        split = get_split(subject_id)
        if split is None or class_id not in LSA64_CLASS_NAMES:
            split_counts["skip"] += 1
            continue
        samples.append((vpath, class_id, subject_id, rep_id, split))
        split_counts[split] += 1

    print(f"\nDistribución por split:")
    print(f"  Train : {split_counts['train']} videos (sujetos 1-6)")
    print(f"  Val   : {split_counts['val']} videos (sujetos 7-8)")
    print(f"  Test  : {split_counts['test']} videos (sujetos 9-10)")
    print(f"  Skip  : {split_counts['skip']} videos ignorados")
    print(f"\nWorkers: {args.num_workers} (de {cpu_count()} núcleos lógicos)")

    if args.dry_run:
        print("\n[DRY RUN] No se procesaron videos.")
        return

    # ── Setup inicial (modelo + carpetas + JSON) ──────────────────────────────
    ensure_holistic_model()

    for split in ["train", "val", "test"]:
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    class_to_idx = {
        LSA64_CLASS_NAMES[i]: i - 1
        for i in range(1, 65)
    }
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    mapping_path = output_dir / "class_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(class_to_idx, f, indent=2)
    print(f"class_mapping.json guardado: {mapping_path}\n")

    # ── Construir lista de tareas ─────────────────────────────────────────────
    tasks = [
        (vpath, class_id, subject_id, rep_id, split,
         str(output_dir), args.max_seq_len, class_to_idx)
        for vpath, class_id, subject_id, rep_id, split in samples
    ]

    # ── Procesar con Pool ─────────────────────────────────────────────────────
    counters = {"ok": 0, "skip": 0, "error": 0, "empty": 0}
    total    = len(tasks)
    t0       = time.time()

    # Intentar importar tqdm para barra de progreso (opcional)
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False
        print("Tip: instala tqdm para barra de progreso → pip install tqdm")

    print(f"Iniciando extracción de {total} videos...\n")

    with Pool(
        processes=args.num_workers,
        initializer=_worker_init,
        initargs=(HOLISTIC_MODEL_PATH,),
    ) as pool:
        iterator = pool.imap_unordered(_process_sample, tasks, chunksize=4)

        if use_tqdm:
            iterator = tqdm(iterator, total=total, unit="vid",
                            desc="Extrayendo", dynamic_ncols=True)

        for i, result in enumerate(iterator, 1):
            status = result["status"]
            counters[status] = counters.get(status, 0) + 1

            if status == "error":
                msg = f"  ERROR [{result['name']}]: {result.get('error','?')}"
                print(msg, flush=True)
            elif status == "ok" and (i % 200 == 0 or i == total):
                elapsed = time.time() - t0
                speed   = i / elapsed
                eta_s   = (total - i) / speed if speed > 0 else 0
                print(
                    f"  [{i}/{total}] {result['split']}/{result['name']} "
                    f"| {result['class']} | {result['frames']}f "
                    f"| {speed:.1f} vid/s | ETA {eta_s/60:.1f} min",
                    flush=True
                )

    elapsed_total = time.time() - t0
    print(f"\n{'─'*60}")
    print(f"✓ Extracción completada en {elapsed_total/60:.1f} min")
    print(f"  OK     : {counters['ok']}")
    print(f"  Saltado: {counters['skip']} (ya existían)")
    print(f"  Vacío  : {counters['empty']}")
    print(f"  Error  : {counters['error']}")
    print(f"  Salida : {output_dir}")

    # ── Guardar resumen ───────────────────────────────────────────────────────
    summary = {
        "dataset":     "LSA64",
        "num_classes": 64,
        "class_to_idx": class_to_idx,
        "idx_to_class": {str(k): v for k, v in idx_to_class.items()},
        "split": {
            "train_subjects": sorted(TRAIN_SUBJECTS),
            "val_subjects":   sorted(VAL_SUBJECTS),
            "test_subjects":  sorted(TEST_SUBJECTS),
        },
        "counts":  split_counts,
        "results": counters,
        "workers": args.num_workers,
        "elapsed_min": round(elapsed_total / 60, 2),
    }
    summary_path = output_dir / "lsa64_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Resumen: {summary_path}")


if __name__ == "__main__":
    main()
