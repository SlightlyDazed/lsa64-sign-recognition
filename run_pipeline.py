"""
run_pipeline.py — Script maestro que ejecuta todas las etapas del pipeline.

Uso:
    python run_pipeline.py --stages all                    # Pipeline completo
    python run_pipeline.py --stages 1,2                    # Solo extracción + preprocesamiento
    python run_pipeline.py --stages 4,5,6                  # Solo entrenamiento
    python run_pipeline.py --stages 2 --dataset autsl      # Preprocesar solo AUTSL
    python run_pipeline.py --stages 7,8                    # Evaluar y exportar
    python run_pipeline.py --stages all --max-samples 50   # Pipeline rápido (debug)
"""
import os
import sys
import argparse
import subprocess
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

STAGES = {
    1: {
        "name": "Descompresión de datasets",
        "script": "01_extract_datasets.py",
        "args": [],
    },
    2: {
        "name": "Preprocesamiento (frames + keypoints)",
        "script": "02_preprocess_frames.py",
        "args": ["--dataset", "{dataset}"],
    },
    3: {
        "name": "Conversión a formato YOLO (opcional)",
        "script": "03_convert_to_yolo.py",
        "args": ["--dataset", "{dataset}"],
    },
    4: {
        "name": "Pre-entrenamiento con WLASL",
        "script": "04_train_wlasl.py",
        "args": [],
    },
    5: {
        "name": "Fine-tuning con AUTSL",
        "script": "05_finetune_autsl.py",
        "args": [],
    },
    6: {
        "name": "Fine-tuning final con LSP PUCP",
        "script": "06_finetune_lsp.py",
        "args": [],
    },
    7: {
        "name": "Evaluación del modelo",
        "script": "07_evaluate.py",
        "args": [],
    },
    8: {
        "name": "Exportación (ONNX + TorchScript)",
        "script": "08_export_model.py",
        "args": [],
    },
}


def run_stage(stage_num: int, extra_args: list, python_exe: str):
    """Ejecuta una etapa del pipeline."""
    stage = STAGES[stage_num]
    script = os.path.join(config.PROJECT_ROOT, stage.get("script"))

    if not os.path.exists(script):
        print(f"  ERROR: Script no encontrado: {script}")
        return False

    cmd = [python_exe, script] + extra_args

    print(f"\n{'='*60}")
    print(f"  ETAPA {stage_num}: {stage['name']}")
    print(f"  Comando: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    start = time.time()
    result = subprocess.run(cmd, cwd=config.PROJECT_ROOT)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n  ✗ Etapa {stage_num} falló (código {result.returncode})")
        print(f"  Tiempo: {elapsed:.1f}s")
        return False

    print(f"\n  ✓ Etapa {stage_num} completada en {elapsed:.1f}s")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline completo de reconocimiento de señas"
    )
    parser.add_argument("--stages", type=str, default="all",
                        help="Etapas a ejecutar: 'all' o números separados "
                             "por coma (ej: '1,2,4,5,6')")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["wlasl", "autsl", "lsp_pucp", "all"],
                        help="Dataset para etapa 2 y 3")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limitar muestras por split (para debug)")
    parser.add_argument("--max-videos", type=int, default=None,
                        help="Limitar videos en preprocesamiento")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override de epochs para entrenamiento")
    parser.add_argument("--python", type=str, default=sys.executable,
                        help="Ruta al ejecutable de Python")
    parser.add_argument("--skip-errors", action="store_true",
                        help="Continuar con la siguiente etapa si una falla")
    args = parser.parse_args()

    # Parsear etapas
    if args.stages == "all":
        stages_to_run = list(STAGES.keys())
    else:
        stages_to_run = [int(s.strip()) for s in args.stages.split(",")]

    print("╔══════════════════════════════════════════════════════════╗")
    print("║   PIPELINE DE RECONOCIMIENTO DE LENGUA DE SEÑAS        ║")
    print("║   YOLO-Pose + BiLSTM + Transfer Learning               ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"\nEtapas: {stages_to_run}")
    print(f"Dataset: {args.dataset}")
    print(f"Python: {args.python}")
    print(f"Device: {config.DEVICE}")

    total_start = time.time()
    results = {}

    for stage_num in stages_to_run:
        if stage_num not in STAGES:
            print(f"Etapa {stage_num} no existe, saltando...")
            continue

        # Construir args extra
        extra_args = []
        stage = STAGES[stage_num]

        # Reemplazar {dataset} en args
        final_args = []
        for a in stage.get("args", []):
            final_args.append(a.replace("{dataset}", args.dataset))
        extra_args.extend(final_args)

        # Agregar flags comunes
        if args.max_samples and stage_num in (4, 5, 6, 7):
            extra_args.extend(["--max-samples", str(args.max_samples)])
        if args.max_videos and stage_num in (2, 3):
            extra_args.extend(["--max-videos", str(args.max_videos)])
        if args.epochs and stage_num in (4, 5, 6):
            extra_args.extend(["--epochs", str(args.epochs)])

        success = run_stage(stage_num, extra_args, args.python)
        results[stage_num] = success

        if not success and not args.skip_errors:
            print(f"\nPipeline detenido en etapa {stage_num}.")
            print("Usa --skip-errors para continuar a pesar de errores.")
            break

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"RESUMEN DEL PIPELINE")
    print(f"{'='*60}")
    for stage_num, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} Etapa {stage_num}: {STAGES[stage_num]['name']}")
    print(f"\nTiempo total: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
