"""
05_finetune_autsl.py — Fine-tuning intermedio con AUTSL (226 clases, lengua de señas turca).

Carga el modelo pre-entrenado con WLASL, reemplaza la cabeza de clasificación,
y entrena con backbone parcialmente congelado.

Uso:
    python 05_finetune_autsl.py [--epochs 30] [--max-samples 100]
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from model import SignClassifier
from dataset_loader import get_dataloaders
from train_utils import train_model
from utils import load_checkpoint


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tuning con AUTSL (226 clases)")
    parser.add_argument("--epochs", type=int,
                        default=config.AUTSL_TRAIN["epochs"])
    parser.add_argument("--batch-size", type=int,
                        default=config.AUTSL_TRAIN["batch_size"])
    parser.add_argument("--lr", type=float,
                        default=config.AUTSL_TRAIN["lr"])
    parser.add_argument("--freeze-epochs", type=int,
                        default=config.AUTSL_TRAIN["freeze_epochs"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--from-scratch", action="store_true",
                        help="Entrenar desde cero (sin transfer learning)")
    parser.add_argument("--resume", action="store_true",
                        help="Continuar entrenamiento desde el último checkpoint guardado")
    parser.add_argument("--extra-epochs", type=int, default=50,
                        help="Epochs adicionales al reanudar (default: 50)")
    parser.add_argument("--resume-lr", type=float, default=3e-5,
                        help="LR inicial para el scheduler fresco al resumir (default: 3e-5)")
    args = parser.parse_args()

    # Advertir combinaciones inválidas de flags
    if args.from_scratch and args.freeze_epochs > 0:
        print("ADVERTENCIA: --freeze-epochs no tiene efecto con --from-scratch. "
              "Se ignorará el congelamiento del backbone.")
    if args.resume and args.from_scratch:
        print("ERROR: --resume y --from-scratch son incompatibles.")
        sys.exit(1)

    device = config.DEVICE

    # Cargar datos AUTSL
    print("Cargando datos AUTSL...")
    loaders = get_dataloaders(
        "autsl",
        batch_size=args.batch_size,
        max_seq_len=config.MAX_SEQ_LEN,
        max_samples=args.max_samples,
    )

    num_classes = loaders["num_classes"]
    if num_classes == 0:
        print("ERROR: No se encontraron datos de AUTSL.")
        print("Ejecuta primero:")
        print("  python 01_extract_datasets.py")
        print("  python 02_preprocess_frames.py --dataset autsl")
        sys.exit(1)

    print(f"Clases AUTSL: {num_classes}, Input dim: {config.INPUT_DIM}")

    wlasl_num_classes = config.WLASL_TRAIN.get("num_classes", 100)
    autsl_ckpt = config.AUTSL_TRAIN["checkpoint"]

    # ---------------------------------------------------------------
    # MODO RESUME: recargar checkpoint AUTSL y seguir entrenando
    # con un scheduler cosine fresco. El LR en epoch 30 era ~1e-7
    # (scheduler agotado), por eso el modelo dejó de mejorar.
    # Con resume_lr=3e-5 y extra_epochs=50 puede explorar más.
    # ---------------------------------------------------------------
    if args.resume:
        if not os.path.exists(autsl_ckpt):
            print(f"ERROR: No se encontró checkpoint en {autsl_ckpt}")
            sys.exit(1)

        model = SignClassifier(
            input_dim=config.INPUT_DIM,
            hidden_dim=config.MODEL_HIDDEN_DIM,
            num_layers=config.MODEL_NUM_LSTM_LAYERS,
            num_classes=num_classes,
            dropout=config.MODEL_DROPOUT,
            bidirectional=config.MODEL_BIDIRECTIONAL,
        )
        print(f"Cargando checkpoint AUTSL para continuar: {autsl_ckpt}")
        start_epoch, prev_metrics = load_checkpoint(model, autsl_ckpt, device=device)
        print(f"  Retomando desde epoch {start_epoch}, "
              f"val_acc previa: {prev_metrics.get('val_acc', '?'):.4f}")
        model = model.to(device)

        train_config = {
            "epochs": args.extra_epochs,
            "lr": args.resume_lr,         # LR fresco — scheduler cosine nuevo
            "weight_decay": config.AUTSL_TRAIN["weight_decay"],
        }

    # ---------------------------------------------------------------
    # MODO NORMAL: entrenamiento desde WLASL o desde cero
    # ---------------------------------------------------------------
    else:
        model = SignClassifier(
            input_dim=config.INPUT_DIM,
            hidden_dim=config.MODEL_HIDDEN_DIM,
            num_layers=config.MODEL_NUM_LSTM_LAYERS,
            num_classes=wlasl_num_classes,
            dropout=config.MODEL_DROPOUT,
            bidirectional=config.MODEL_BIDIRECTIONAL,
        )

        wlasl_ckpt = config.WLASL_TRAIN["checkpoint"]
        if os.path.exists(wlasl_ckpt) and not args.from_scratch:
            print(f"Cargando pesos pre-entrenados: {wlasl_ckpt}")
            epoch, pretrain_metrics = load_checkpoint(model, wlasl_ckpt, device=device)
            print(f"  Checkpoint WLASL epoch {epoch}, metrics: {pretrain_metrics}")
        else:
            if not args.from_scratch:
                print("ADVERTENCIA: No se encontró checkpoint de WLASL.")
                print("Entrenando desde cero.")

        model.replace_head(num_classes)
        model = model.to(device)

        if args.freeze_epochs > 0 and not args.from_scratch:
            model.freeze_backbone(freeze_input_proj=True, freeze_temporal=False)

        train_config = {
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": config.AUTSL_TRAIN["weight_decay"],
        }

    # FIX #4: Envolver train_model en try/except para manejar fallos
    # y evitar que metrics quede indefinido al guardar el JSON.
    try:
        metrics = train_model(
            model=model,
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            train_config=train_config,
            device=device,
            checkpoint_path=config.AUTSL_TRAIN["checkpoint"],
            results_prefix="autsl",
            freeze_epochs=args.freeze_epochs,
        )
    except Exception as e:
        print(f"ERROR durante el entrenamiento: {e}")
        sys.exit(1)

    # Guardar info de clases
    class_info_path = os.path.join(config.MODELS_DIR, "autsl_classes.json")
    with open(class_info_path, "w") as f:
        json.dump({
            "class_to_idx": loaders["class_to_idx"],
            "num_classes": num_classes,
            "metrics": {
                "best_val_acc": metrics.get("best_val_acc"),
                "final_train_acc": metrics.get("final_train_acc"),
            }
        }, f, indent=2)

    print(f"Info de clases guardada en {class_info_path}")
    print("✓ Etapa 5 (AUTSL fine-tuning) completada.")


if __name__ == "__main__":
    main()
