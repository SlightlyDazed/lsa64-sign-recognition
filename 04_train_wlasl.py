"""
04_train_wlasl.py — Pre-entrenamiento del clasificador temporal con WLASL (top-100 clases ASL).

Uso:
    python 04_train_wlasl.py [--epochs 50] [--batch-size 64] [--max-samples 100]
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


def main():
    parser = argparse.ArgumentParser(
        description="Pre-entrena el clasificador con WLASL (top-100)")
    parser.add_argument("--epochs", type=int,
                        default=config.WLASL_TRAIN["epochs"])
    parser.add_argument("--batch-size", type=int,
                        default=config.WLASL_TRAIN["batch_size"])
    parser.add_argument("--lr", type=float,
                        default=config.WLASL_TRAIN["lr"])
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limitar muestras (para debug rápido)")
    parser.add_argument("--use-transformer", action="store_true",
                        help="Usar Transformer en vez de LSTM")
    parser.add_argument("--freeze-epochs", type=int, default=0,
                        help="Epochs con capas congeladas al inicio (0 = sin freeze)")
    args = parser.parse_args()

    device = config.DEVICE

    # Cargar datos
    print("Cargando datos WLASL...")
    loaders = get_dataloaders(
        "wlasl",
        batch_size=args.batch_size,
        max_seq_len=config.MAX_SEQ_LEN,
        max_samples=args.max_samples,
    )

    num_classes = loaders["num_classes"]
    if num_classes == 0:
        print("ERROR: No se encontraron datos de WLASL.")
        print("Asegúrate de ejecutar primero:")
        print("  python 01_extract_datasets.py")
        print("  python 02_preprocess_frames.py --dataset wlasl")
        sys.exit(1)

    print(f"  Clases:  {num_classes}")
    print(f"  Train:   {len(loaders['train'].dataset)} muestras")
    print(f"  Val:     {len(loaders['val'].dataset)} muestras")
    print(f"  Input dim: {config.INPUT_DIM}")

    # Crear modelo
    if args.use_transformer:
        from model import SignClassifierTransformer
        model = SignClassifierTransformer(
            input_dim=config.INPUT_DIM,
            hidden_dim=config.MODEL_HIDDEN_DIM,
            num_classes=num_classes,
            dropout=config.MODEL_DROPOUT,
            max_seq_len=config.MAX_SEQ_LEN,
        )
    else:
        model = SignClassifier(
            input_dim=config.INPUT_DIM,
            hidden_dim=config.MODEL_HIDDEN_DIM,
            num_layers=config.MODEL_NUM_LSTM_LAYERS,
            num_classes=num_classes,
            dropout=config.MODEL_DROPOUT,
            bidirectional=config.MODEL_BIDIRECTIONAL,
        )

    model = model.to(device)

    # Entrenar
    train_config = {
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": config.WLASL_TRAIN["weight_decay"],
    }

    # Checkpoint versionado por tipo de modelo: evita sobreescribir al comparar LSTM vs Transformer
    base_ckpt = config.WLASL_TRAIN["checkpoint"]
    ckpt_stem = os.path.splitext(base_ckpt)[0]
    ckpt_ext  = os.path.splitext(base_ckpt)[1] or ".pt"
    model_tag = "transformer" if args.use_transformer else "lstm"
    checkpoint_path = f"{ckpt_stem}_{model_tag}{ckpt_ext}"

    metrics = train_model(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        train_config=train_config,
        device=device,
        checkpoint_path=checkpoint_path,
        results_prefix="wlasl",
        freeze_epochs=args.freeze_epochs,
    )

    # Guardar info de clases
    class_info_path = os.path.join(config.MODELS_DIR, "wlasl_classes.json")
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    with open(class_info_path, "w") as f:
        json.dump({
            "class_to_idx": loaders["class_to_idx"],
            "num_classes": num_classes,
            "model": model_tag,
            "checkpoint": checkpoint_path,
            "metrics": {
                "best_val_acc":    metrics.get("best_val_acc"),
                "final_train_acc": metrics.get("final_train_acc"),
            }
        }, f, indent=2)

    print(f"Info de clases guardada en {class_info_path}")
    print("✓ Etapa 4 (WLASL pre-training) completada.")


if __name__ == "__main__":
    main()
