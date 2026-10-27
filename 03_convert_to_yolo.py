"""
03_convert_to_yolo.py — Convierte anotaciones a formato YOLO-cls.
Genera la estructura de directorios que ultralytics espera para
clasificación de imágenes (opcional, para usar YOLO directamente).

Este script es opcional — el pipeline principal usa el clasificador
temporal LSTM/Transformer sobre secuencias de keypoints.

Uso:
    python 03_convert_to_yolo.py --dataset autsl
"""
import os
import sys
import csv
import argparse
import cv2
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from utils import setup_logger

logger = setup_logger("convert_yolo",
                      os.path.join(config.RESULTS_DIR, "03_convert.log"))


def create_yolo_cls_structure(dataset_name: str, max_videos: int = None):
    """
    Crea estructura de directorios para YOLO clasificación.
    Extrae el frame central de cada video como imagen representativa.

    Estructura:
        data/yolo_cls/{dataset}/train/class_name/video_id.jpg
        data/yolo_cls/{dataset}/val/class_name/video_id.jpg
        data/yolo_cls/{dataset}/test/class_name/video_id.jpg
    """
    yolo_dir = os.path.join(config.DATA_DIR, "yolo_cls", dataset_name)

    if dataset_name == "autsl":
        _convert_autsl_yolo_cls(yolo_dir, max_videos)
    elif dataset_name == "lsp_pucp":
        _convert_lsp_yolo_cls(yolo_dir, max_videos)
    elif dataset_name == "wlasl":
        _convert_wlasl_yolo_cls(yolo_dir, max_videos)


def _convert_autsl_yolo_cls(yolo_dir: str, max_videos: int):
    """Convierte AUTSL a formato YOLO-cls."""
    autsl_root = config.RAW_AUTSL
    csv_files = list(Path(autsl_root).rglob("train.csv"))
    if not csv_files:
        logger.error("train.csv no encontrado")
        return
    base_dir = csv_files[0].parent

    for split_name, csv_name in [("train", "train.csv"),
                                  ("val", "val.csv"),
                                  ("test", "test.csv")]:
        csv_path = base_dir / csv_name
        if not csv_path.exists():
            continue

        with open(csv_path) as f:
            rows = list(csv.reader(f))

        created, skipped, errors = 0, 0, 0
        for row in tqdm(rows, desc=f"YOLO-cls AUTSL {split_name}"):
            if max_videos and (created + skipped) >= max_videos:
                break
            if len(row) < 2:
                continue

            filename, label_id = row[0].strip(), row[1].strip()
            class_name = f"class_{int(label_id):03d}"

            video_path = None
            for candidate_dir in [base_dir / split_name, base_dir]:
                for ext in ["", ".mp4"]:
                    p = candidate_dir / (filename + ext)
                    if p.exists():
                        video_path = str(p)
                        break
                if video_path:
                    break

            if not video_path:
                continue

            out_path = os.path.join(yolo_dir, split_name, class_name,
                                    f"{Path(filename).stem}.jpg")
            if os.path.exists(out_path):
                skipped += 1
                continue

            try:
                cap = cv2.VideoCapture(video_path)
                try:
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    # CAP_PROP_FRAME_COUNT devuelve -1 o 0 con algunos codecs
                    mid = max(0, total_frames // 2) if total_frames > 0 else 0
                    cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
                    ret, frame = cap.read()
                finally:
                    cap.release()  # siempre liberar, incluso si hay excepción

                if ret:
                    frame = cv2.resize(frame, config.FRAME_SIZE)
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    cv2.imwrite(out_path, frame)
                    created += 1
            except Exception as e:
                logger.warning(f"Error en {filename}: {e}")
                errors += 1

        logger.info(f"AUTSL {split_name}: {created} creadas, {skipped} ya existían, {errors} errores")


def _convert_lsp_yolo_cls(yolo_dir: str, max_videos: int):
    """Convierte LSP-PUCP a formato YOLO-cls."""
    seq_dir = os.path.join(config.PROCESSED_LSP, "sequences")
    if not os.path.isdir(seq_dir):
        logger.error("Primero ejecutar 02_preprocess_frames.py para LSP PUCP")
        return

    lsp_root = config.RAW_LSP
    seg_dirs = list(Path(lsp_root).rglob("SEGMENTED_SIGN_ADJUSTED"))
    if not seg_dirs:
        return
    seg_dir_path = seg_dirs[0]

    # FIX: índice pre-construido — evita rglob O(n²) (un scan por video)
    logger.info("  Indexando videos LSP...")
    video_index = {p.stem: p for p in seg_dir_path.rglob("*.mp4")}
    logger.info(f"  {len(video_index)} videos indexados")

    for split_name in ["train", "val", "test"]:
        split_dir = os.path.join(seq_dir, split_name)
        if not os.path.isdir(split_dir):
            continue

        created, skipped, errors = 0, 0, 0
        total_processed = 0
        stop = False  # FIX: flag para romper el loop externo correctamente

        for class_name in sorted(os.listdir(split_dir)):
            if stop:
                break
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            for npy_file in os.listdir(class_dir):
                # FIX: chequeo en el loop interno, con flag para salir del externo
                if max_videos and total_processed >= max_videos:
                    stop = True
                    break
                if not npy_file.endswith(".npy"):
                    continue

                video_stem = npy_file.replace(".npy", "")
                # FIX: lookup O(1) en el índice en vez de rglob por archivo
                video_path = video_index.get(video_stem)
                if not video_path:
                    continue

                out_path = os.path.join(yolo_dir, split_name, class_name,
                                        f"{video_stem}.jpg")
                if os.path.exists(out_path):
                    skipped += 1
                    total_processed += 1
                    continue

                try:
                    cap = cv2.VideoCapture(str(video_path))
                    try:
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        mid = max(0, total_frames // 2) if total_frames > 0 else 0
                        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
                        ret, frame = cap.read()
                    finally:
                        cap.release()  # siempre liberar

                    if ret:
                        frame = cv2.resize(frame, config.FRAME_SIZE)
                        os.makedirs(os.path.dirname(out_path), exist_ok=True)
                        cv2.imwrite(out_path, frame)
                        created += 1
                        total_processed += 1
                except Exception as e:
                    logger.warning(f"Error: {e}")
                    errors += 1

        logger.info(f"LSP {split_name}: {created} creadas, {skipped} ya existían, {errors} errores")


def _convert_wlasl_yolo_cls(yolo_dir: str, max_videos: int):
    """Convierte WLASL a formato YOLO-cls (si hay videos locales)."""
    logger.info("WLASL YOLO-cls: skipped (requiere videos locales)")


def main():
    parser = argparse.ArgumentParser(
        description="Convierte anotaciones a formato YOLO-cls"
    )
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["wlasl", "autsl", "lsp_pucp", "all"])
    parser.add_argument("--max-videos", type=int, default=None)
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset != "all" else ["wlasl", "autsl", "lsp_pucp"]
    for ds in datasets:
        create_yolo_cls_structure(ds, args.max_videos)

    logger.info("✓ Conversión a formato YOLO completada.")


if __name__ == "__main__":
    main()
