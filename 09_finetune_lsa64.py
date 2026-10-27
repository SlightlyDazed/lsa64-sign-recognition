"""
09_finetune_lsa64.py — Fine-tuning con LSA64 (64 clases, lengua de señas argentina).

Carga el modelo pre-entrenado con AUTSL (226 clases), reemplaza la cabeza
de clasificación, y entrena con backbone parcialmente congelado.

Uso:
    python 09_finetune_lsa64.py [--epochs 40] [--freeze-epochs 5]
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

LSA64_NUM_CLASSES = 64


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tuning con LSA64 (64 clases)")
    parser.add_argument("--epochs",        type=int,   default=40)
    parser.add_argument("--batch-size",    type=int,   default=32)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--freeze-epochs", type=int,   default=5,
                        help="Epochs con backbone congelado antes de descongelar todo")
    parser.add_argument("--weight-decay",  type=float, default=1e-4)
    parser.add_argument("--max-samples",   type=int,   default=None)
    parser.add_argument("--from-scratch",  action="store_true",
                        help="Entrenar desde cero sin transfer learning")
    parser.add_argument("--checkpoint-out", type=str,
                        default=os.path.join(config.MODELS_DIR, "lsa64_finetuned.pth"),
                        help="Ruta donde guardar el checkpoint final")
    parser.add_argument("--resume", action="store_true",
                        help="Reanudar entrenamiento desde --checkpoint-out en vez de cargar AUTSL")
    args = parser.parse_args()

    # Validar flags contradictorios
    if args.from_scratch and args.freeze_epochs > 0:
        print("ADVERTENCIA: --freeze-epochs no tiene efecto con --from-scratch. "
              "Se ignorará el congelamiento.")

    device = config.DEVICE

    # Cargar mapeo de clases LSA64
    lsa64_dir      = os.path.join(config.PROCESSED_DIR, "lsa64")
    class_map_path = os.path.join(lsa64_dir, "class_mapping.json")

    class_to_idx = None
    if os.path.exists(class_map_path):
        with open(class_map_path) as f:
            class_to_idx = json.load(f)
        print(f"Mapeo de clases cargado: {len(class_to_idx)} clases")
    else:
        print("ADVERTENCIA: No se encontró class_mapping.json.")
        print(f"Ejecuta primero: python 08_extract_lsa64.py")

    # Cargar datos LSA64
    print("Cargando datos LSA64...")
    loaders = get_dataloaders(
        "lsa64",
        batch_size=args.batch_size,
        max_seq_len=config.MAX_SEQ_LEN,
        max_samples=args.max_samples,
        augment_train=True,
        class_to_idx=class_to_idx,
    )

    num_classes = loaders["num_classes"]
    if num_classes == 0:
        print("ERROR: No se encontraron datos de LSA64.")
        print("Ejecuta primero: python 08_extract_lsa64.py")
        sys.exit(1)

    print(f"Clases LSA64: {num_classes}, Input dim: {config.INPUT_DIM}")

    # Determinar num_classes del checkpoint fuente para crear arquitectura compatible
    autsl_ckpt = config.AUTSL_TRAIN["checkpoint"]
    if args.resume:
        # Resumiendo LSA64: el checkpoint ya tiene cabeza de 64 clases
        import torch
        ckpt_tmp = torch.load(args.checkpoint_out, map_location="cpu")
        hidden_dim_ckpt   = ckpt_tmp["model_state_dict"]["input_proj.0.weight"].shape[0]
        autsl_num_classes = num_classes  # cabeza ya es LSA64
        print(f"  Arquitectura detectada del checkpoint: hidden_dim={hidden_dim_ckpt}")
        del ckpt_tmp
    elif not args.from_scratch:
        autsl_num_classes = config.AUTSL_TRAIN.get("num_classes", 226)

        # Inferir hidden_dim desde el checkpoint para evitar size mismatch
        if os.path.exists(autsl_ckpt):
            import torch
            ckpt_tmp = torch.load(autsl_ckpt, map_location="cpu")
            hidden_dim_ckpt = ckpt_tmp["model_state_dict"]["input_proj.0.weight"].shape[0]
            print(f"  Arquitectura detectada del checkpoint: hidden_dim={hidden_dim_ckpt}")
            del ckpt_tmp  # liberar memoria
        else:
            hidden_dim_ckpt = config.MODEL_HIDDEN_DIM
    else:
        autsl_num_classes = num_classes  # from-scratch: directo a 64 clases
        hidden_dim_ckpt   = config.MODEL_HIDDEN_DIM

    # Crear modelo con arquitectura compatible con el checkpoint AUTSL
    model = SignClassifier(
        input_dim=config.INPUT_DIM,
        hidden_dim=hidden_dim_ckpt,     # consistente con el checkpoint cargado
        num_layers=config.MODEL_NUM_LSTM_LAYERS,
        num_classes=autsl_num_classes,  # consistente con checkpoint fuente
        dropout=config.MODEL_DROPOUT,
        bidirectional=config.MODEL_BIDIRECTIONAL,
    )

    # Cargar pesos: desde checkpoint LSA64 (resume) o desde AUTSL (transfer)
    if args.resume:
        if os.path.exists(args.checkpoint_out):
            print(f"Reanudando desde checkpoint LSA64: {args.checkpoint_out}")
            epoch, prev_metrics = load_checkpoint(model, args.checkpoint_out, device=device)
            print(f"  Reanudando desde epoch {epoch}, metrics: {prev_metrics}")
        else:
            print(f"ADVERTENCIA: No se encontró checkpoint en {args.checkpoint_out}. Cargando AUTSL.")
            args.resume = False

    if not args.from_scratch and not args.resume:
        if os.path.exists(autsl_ckpt):
            print(f"Cargando pesos AUTSL: {autsl_ckpt}")
            epoch, pretrain_metrics = load_checkpoint(model, autsl_ckpt, device=device)
            print(f"  Checkpoint AUTSL epoch {epoch}, metrics: {pretrain_metrics}")
        else:
            print(f"ADVERTENCIA: No se encontró checkpoint AUTSL en {autsl_ckpt}.")
            print("Entrenando desde cero.")

    # Reemplazar cabeza para LSA64 (solo si NO se está reanudando)
    if not args.resume:
        model.replace_head(num_classes)

    # Mover a device ANTES de congelar
    model = model.to(device)

    # Congelar backbone los primeros N epochs
    if args.freeze_epochs > 0 and not args.from_scratch:
        model.freeze_backbone(freeze_input_proj=True, freeze_temporal=True)

    # Resumen de parámetros
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParámetros totales:      {total:,}")
    print(f"Parámetros entrenables: {trainable:,}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}, "
          f"Freeze epochs: {args.freeze_epochs}\n")

    # Entrenar
    train_config = {
        "epochs":       args.epochs,
        "lr":           args.lr,
        "weight_decay": args.weight_decay,
    }

    try:
        metrics = train_model(
            model=model,
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            train_config=train_config,
            device=device,
            checkpoint_path=args.checkpoint_out,
            results_prefix="lsa64",
            freeze_epochs=args.freeze_epochs,
        )
    except Exception as e:
        print(f"ERROR durante el entrenamiento: {e}")
        sys.exit(1)

    # Guardar info de clases y métricas, PREVINIENDO bug de 'KeyError' si metrics falla
    class_info_path = os.path.join(config.MODELS_DIR, "lsa64_classes.json")
    with open(class_info_path, "w") as f:
        json.dump({
            "dataset":       "LSA64",
            "num_classes":   num_classes,
            "class_to_idx":  loaders["class_to_idx"],
            "idx_to_class":  loaders["idx_to_class"],
            "pretrained_on": "AUTSL" if not args.from_scratch else "none",
            "metrics": {
                "best_val_acc":    metrics.get("best_val_acc", None),
                "final_train_acc": metrics.get("final_train_acc", None),
            },
        }, f, ensure_ascii=False, indent=2)

    print(f"\nInfo de clases guardada en: {class_info_path}")
    print(f"Checkpoint guardado en:     {args.checkpoint_out}")
    print("✓ Fine-tuning LSA64 completado.")


if __name__ == "__main__":
    main()
