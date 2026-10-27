"""
01_extract_datasets.py — Descomprime los 3 datasets a data/raw/.
Idempotente: no re-extrae si los directorios ya existen con contenido.

Uso:
    python 01_extract_datasets.py [--force]
"""
import os
import sys
import zipfile
import tarfile
import argparse
from pathlib import Path

# Asegurar que se puede importar config desde el mismo directorio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from utils import setup_logger

logger = setup_logger("extract", os.path.join(config.RESULTS_DIR, "01_extract.log"))


def dir_has_content(path: str) -> bool:
    """Verifica si un directorio existe y tiene contenido."""
    if not os.path.isdir(path):
        return False
    return len(os.listdir(path)) > 0


def extract_zip(zip_path: str, dest_dir: str, name: str):
    """Extrae un archivo ZIP."""
    if dir_has_content(dest_dir):
        logger.info(f"[{name}] Ya extraído en {dest_dir}, saltando...")
        return

    logger.info(f"[{name}] Extrayendo {zip_path} → {dest_dir}")
    os.makedirs(dest_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zf:
        total = len(zf.namelist())
        for i, member in enumerate(zf.namelist()):
            zf.extract(member, dest_dir)
            if (i + 1) % 5000 == 0:
                logger.info(f"  [{name}] {i+1}/{total} archivos extraídos...")

    logger.info(f"[{name}] ✓ Extracción completa: {total} archivos")


def extract_tar(tar_path: str, dest_dir: str, name: str):
    """Extrae un archivo TAR."""
    if dir_has_content(dest_dir):
        logger.info(f"[{name}] Ya extraído en {dest_dir}, saltando...")
        return

    logger.info(f"[{name}] Extrayendo {tar_path} → {dest_dir}")
    os.makedirs(dest_dir, exist_ok=True)

    with tarfile.open(tar_path, 'r') as tf:
        members = tf.getmembers()
        total = len(members)
        for i, member in enumerate(members):
            tf.extract(member, dest_dir, filter='data')
            if (i + 1) % 2000 == 0:
                logger.info(f"  [{name}] {i+1}/{total} archivos extraídos...")

    logger.info(f"[{name}] ✓ Extracción completa: {total} archivos")


def generate_inventory():
    """Genera un inventario de cada dataset extraído."""
    logger.info("=" * 60)
    logger.info("INVENTARIO DE DATASETS")
    logger.info("=" * 60)

    # WLASL
    wlasl_dir = config.RAW_WLASL
    if os.path.isdir(wlasl_dir):
        # Buscar el JSON principal
        json_paths = list(Path(wlasl_dir).rglob("WLASL_v0.3.json"))
        if json_paths:
            import json
            with open(json_paths[0]) as f:
                data = json.load(f)
            total_instances = sum(len(entry.get("instances", []))
                                  for entry in data)
            logger.info(f"[WLASL] {len(data)} glosses, "
                        f"{total_instances} instancias de video")
        # Buscar nslt JSON files
        nslt_files = list(Path(wlasl_dir).rglob("nslt_*.json"))
        for nf in nslt_files:
            import json
            with open(nf) as f:
                nd = json.load(f)
            logger.info(f"  -> {nf.name}: {len(nd)} entradas")

    # AUTSL
    autsl_dir = config.RAW_AUTSL
    if os.path.isdir(autsl_dir):
        for split in ["train", "val", "test"]:
            csv_path = list(Path(autsl_dir).rglob(f"{split}.csv"))
            if csv_path:
                import csv
                with open(csv_path[0]) as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                num_classes = len(set(r[1] for r in rows if len(r) >= 2))
                logger.info(f"[AUTSL] {split}: {len(rows)} videos, "
                            f"{num_classes} clases")

        # Contar videos mp4
        mp4s = list(Path(autsl_dir).rglob("*_color.mp4"))
        logger.info(f"[AUTSL] Total videos MP4: {len(mp4s)}")

    # LSP PUCP
    lsp_dir = config.RAW_LSP
    if os.path.isdir(lsp_dir):
        seg_dir = list(Path(lsp_dir).rglob("SEGMENTED_SIGN_ADJUSTED"))
        if seg_dir:
            seg_dir = seg_dir[0]
            stories = [d for d in seg_dir.iterdir() if d.is_dir()]
            total_videos = 0
            all_labels = set()
            for story in stories:
                videos = list(story.glob("*.mp4"))
                total_videos += len(videos)
                for v in videos:
                    # Label es el nombre sin el id numérico final
                    label = "_".join(v.stem.rsplit("_", 1)[:-1])
                    if label:
                        all_labels.add(label)

            logger.info(f"[LSP PUCP] {len(stories)} historias, "
                        f"{total_videos} videos segmentados, "
                        f"{len(all_labels)} señas únicas")

        # Keypoints
        pkl_dir = list(Path(lsp_dir).rglob("pkl"))
        if pkl_dir:
            pkls = list(pkl_dir[0].rglob("*.pkl"))
            logger.info(f"[LSP PUCP] {len(pkls)} archivos de keypoints .pkl")

    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Extrae los datasets WLASL, AUTSL y LSP PUCP"
    )
    parser.add_argument("--force", action="store_true",
                        help="Forzar re-extracción incluso si ya existe")
    args = parser.parse_args()

    if args.force:
        logger.info("Modo --force activado: re-extrayendo todo")
        import shutil
        for d in [config.RAW_WLASL, config.RAW_AUTSL, config.RAW_LSP]:
            if os.path.isdir(d):
                shutil.rmtree(d)

    # --- WLASL ---
    logger.info("=" * 60)
    logger.info("ETAPA 1: Descompresión de Datasets")
    logger.info("=" * 60)

    extract_zip(
        config.SOURCE_FILES["wlasl_zip"],
        config.RAW_WLASL,
        "WLASL"
    )

    # --- AUTSL ---
    extract_zip(
        config.SOURCE_FILES["autsl_zip"],
        config.RAW_AUTSL,
        "AUTSL"
    )

    # --- LSP PUCP (3 archivos tar) ---
    for key, name in [
        ("lsp_videos_tar", "LSP-Videos"),
        ("lsp_keypoints_tar", "LSP-Keypoints"),
        ("lsp_srt_tar", "LSP-SRT"),
    ]:
        extract_tar(
            config.SOURCE_FILES[key],
            config.RAW_LSP,
            name
        )

    # --- Inventario ---
    generate_inventory()
    logger.info("✓ Etapa 1 completada.")


if __name__ == "__main__":
    main()
