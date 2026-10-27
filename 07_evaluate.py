"""
07_evaluate.py — Evaluación del modelo final con métricas detalladas.

Genera: mAP, accuracy por clase, top-5 accuracy, matriz de confusión,
y reporte de clasificación completo.

Uso:
    python 07_evaluate.py [--model-path models/autsl_finetuned.pth] [--dataset autsl]
"""
import os
import sys
import json
import argparse
import csv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from model import SignClassifier
from dataset_loader import get_dataloaders
from train_utils import evaluate
from utils import load_checkpoint, plot_confusion_matrix

import torch
import torch.nn as nn


def compute_per_class_accuracy(y_true, y_pred, idx_to_class):
    """Calcula accuracy por clase."""
    from collections import defaultdict
    correct_per_class = defaultdict(int)
    total_per_class = defaultdict(int)

    for yt, yp in zip(y_true, y_pred):
        total_per_class[yt] += 1
        if yt == yp:
            correct_per_class[yt] += 1

    results = []
    for idx in sorted(total_per_class.keys()):
        cls_name = idx_to_class.get(idx, f"class_{idx}")
        total = total_per_class[idx]
        correct = correct_per_class[idx]
        acc = correct / total if total > 0 else 0
        results.append({
            "class_idx": idx,
            "class_name": cls_name,
            "total": total,
            "correct": correct,
            "accuracy": acc,
        })

    return results


def compute_topk_accuracy(logits_list, labels_list, k=5):
    """
    FIX #1 y #5: Calcula top-1 y top-k accuracy a partir de logits y labels
    ya recolectados, evitando un segundo recorrido del dataloader.
    """
    logits = torch.cat(logits_list, dim=0)   # (N, num_classes)
    labels = torch.cat(labels_list, dim=0)   # (N,)

    _, topk_preds = logits.topk(min(k, logits.size(1)), dim=1)

    correct_top1 = (topk_preds[:, 0] == labels).sum().item()
    correct_topk = sum(
        labels[i].item() in topk_preds[i].tolist()
        for i in range(len(labels))
    )
    total = len(labels)

    return correct_top1 / max(total, 1), correct_topk / max(total, 1)


def collect_logits(model, dataloader, device):
    """
    FIX #5: Recorre el dataloader UNA sola vez y devuelve logits, preds y labels.
    Reemplaza el doble recorrido de evaluate() + compute_topk_accuracy().
    """
    model.eval()
    all_logits = []
    all_preds = []
    all_labels = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            seqs = batch["sequence"].to(device)
            labels = batch["label"].to(device)
            lengths = batch["length"].to(device)

            logits = model(seqs, lengths)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)

            preds = logits.argmax(dim=1)
            all_logits.append(logits.cpu())
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / max(len(all_labels), 1)
    return avg_loss, all_preds, all_labels, all_logits


def normalize_idx_to_class(idx_to_class):
    """
    FIX #6: Normaliza las keys de idx_to_class a int, ya que JSON
    deserializa todas las keys como strings.
    """
    return {int(k): v for k, v in idx_to_class.items()}


def main():
    parser = argparse.ArgumentParser(description="Evalúa el modelo final")
    parser.add_argument("--model-path", type=str,
                        default=config.AUTSL_TRAIN["checkpoint"],
                        help="Ruta al checkpoint del modelo")
    parser.add_argument("--dataset", type=str, default="autsl",
                        choices=["wlasl", "autsl", "lsp_pucp", "lsa64"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    device = config.DEVICE

    # Cargar mapeo de clases
    dataset_class_files = {
        "wlasl": os.path.join(config.MODELS_DIR, "wlasl_classes.json"),
        "autsl": os.path.join(config.MODELS_DIR, "autsl_classes.json"),
        "lsp_pucp": os.path.join(config.MODELS_DIR, "lsp_classes.json"),
        "lsa64":    os.path.join(config.MODELS_DIR, "lsa64_classes.json"),
    }

    class_to_idx = None
    class_info_path = dataset_class_files[args.dataset]
    if os.path.exists(class_info_path):
        with open(class_info_path) as f:
            class_info = json.load(f)
        class_to_idx = class_info.get("class_to_idx", None)
    else:
        if args.dataset == "lsp_pucp":
            map_path = os.path.join(config.PROCESSED_LSP, "class_mapping.json")
            if os.path.exists(map_path):
                with open(map_path) as f:
                    class_to_idx = json.load(f)

    # Cargar datos
    print(f"Cargando datos de {args.dataset}...")
    loaders = get_dataloaders(
        args.dataset,
        batch_size=args.batch_size,
        max_seq_len=config.MAX_SEQ_LEN,
        augment_train=False,
        max_samples=args.max_samples,
        class_to_idx=class_to_idx,
    )

    num_classes = loaders["num_classes"]

    # FIX #4: Validar test loader ANTES de crear y cargar el modelo
    if loaders["test"] is None:
        print("ERROR: No hay datos de test disponibles.")
        sys.exit(1)

    # FIX #6: Normalizar keys de idx_to_class a int
    idx_to_class = normalize_idx_to_class(loaders["idx_to_class"])
    print(f"Clases: {num_classes}")

    # Verificar checkpoint antes de crear el modelo
    if not os.path.exists(args.model_path):
        print(f"ERROR: Checkpoint no encontrado: {args.model_path}")
        sys.exit(1)

    # Inferir hidden_dim desde el checkpoint para evitar size mismatch
    ckpt_tmp = torch.load(args.model_path, map_location="cpu")
    hidden_dim_ckpt = ckpt_tmp["model_state_dict"]["input_proj.0.weight"].shape[0]
    print(f"  Arquitectura detectada: hidden_dim={hidden_dim_ckpt}")
    del ckpt_tmp

    # Crear y cargar modelo
    model = SignClassifier(
        input_dim=config.INPUT_DIM,
        hidden_dim=hidden_dim_ckpt,
        num_layers=config.MODEL_NUM_LSTM_LAYERS,
        num_classes=num_classes,
        dropout=config.MODEL_DROPOUT,
        bidirectional=config.MODEL_BIDIRECTIONAL,
    )

    print(f"Cargando modelo: {args.model_path}")
    load_checkpoint(model, args.model_path, device=device)
    model = model.to(device)

    # FIX #1 y #5: Un solo recorrido del dataloader para obtener
    # loss, preds, labels y logits — evita el doble recorrido costoso.
    print("Evaluando en test set...")
    test_loss, all_preds, all_labels, all_logits = collect_logits(
        model, loaders["test"], device
    )

    # FIX #1: top1 y top5 calculados desde los mismos logits,
    # consistentes con lo que se imprime y lo que se guarda en JSON.
    top1, top5 = compute_topk_accuracy(all_logits,
                                        [torch.tensor(all_labels)], k=5)

    print(f"\n{'='*60}")
    print(f"RESULTADOS EN TEST SET ({args.dataset})")
    print(f"{'='*60}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy (Top-1): {top1:.4f} ({top1*100:.2f}%)")
    print(f"Top-5 Accuracy: {top5:.4f} ({top5*100:.2f}%)")

    # Accuracy por clase
    per_class = compute_per_class_accuracy(all_labels, all_preds, idx_to_class)

    # Reporte
    print(f"\n{'='*60}")
    print(f"ACCURACY POR CLASE (Top-10 / Bottom-10)")
    print(f"{'='*60}")

    sorted_by_acc = sorted(per_class, key=lambda x: x["accuracy"], reverse=True)

    print("\n--- Top 10 mejores ---")
    for entry in sorted_by_acc[:10]:
        print(f"  {entry['class_name']:30s} | "
              f"Acc: {entry['accuracy']:.2%} | "
              f"({entry['correct']}/{entry['total']})")

    print("\n--- Bottom 10 peores ---")
    for entry in sorted_by_acc[-10:]:
        print(f"  {entry['class_name']:30s} | "
              f"Acc: {entry['accuracy']:.2%} | "
              f"({entry['correct']}/{entry['total']})")

    # FIX #2: Construir class_names_ordered sin saltar índices faltantes,
    # usando un fallback explícito para que siempre haya num_classes entradas
    # y la matriz de confusión no quede desalineada.
    class_names_ordered = [
        idx_to_class.get(i, f"class_{i}") for i in range(num_classes)
    ]

    cm_path = os.path.join(config.RESULTS_DIR,
                           f"{args.dataset}_confusion_matrix.png")
    plot_confusion_matrix(
        [idx_to_class.get(l, str(l)) for l in all_labels],
        [idx_to_class.get(p, str(p)) for p in all_preds],
        class_names_ordered,
        cm_path,
        top_n=30 if len(class_names_ordered) > 30 else None
    )

    # Classification report
    from sklearn.metrics import classification_report

    # FIX #3: Renombrar variable del loop a cls_metrics para evitar
    # colisión con cualquier variable 'metrics' del scope superior.
    report = classification_report(
        all_labels, all_preds,
        target_names=[idx_to_class.get(i, f"cls_{i}")
                      for i in range(num_classes)],
        output_dict=True,
        zero_division=0
    )

    # Guardar reporte como CSV
    report_path = os.path.join(config.RESULTS_DIR,
                               f"{args.dataset}_classification_report.csv")
    with open(report_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1-score", "support"])
        for cls_name, cls_metrics in report.items():   # FIX #3
            if isinstance(cls_metrics, dict):
                writer.writerow([
                    cls_name,
                    f"{cls_metrics['precision']:.4f}",
                    f"{cls_metrics['recall']:.4f}",
                    f"{cls_metrics['f1-score']:.4f}",
                    cls_metrics["support"]
                ])

    print(f"\nReporte guardado en: {report_path}")
    print(f"Matriz de confusión guardada en: {cm_path}")

    # Guardar resumen en JSON — FIX #1: top1 consistente con lo impreso
    summary = {
        "dataset": args.dataset,
        "model_path": args.model_path,
        "num_classes": num_classes,
        "test_loss": test_loss,
        "test_accuracy_top1": top1,
        "test_accuracy_top5": top5,
        "per_class_accuracy": per_class,
        "macro_avg": report.get("macro avg", {}),
        "weighted_avg": report.get("weighted avg", {}),
    }

    summary_path = os.path.join(config.RESULTS_DIR,
                                f"{args.dataset}_evaluation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Resumen JSON: {summary_path}")
    print(f"\n✓ Evaluación completada.")


if __name__ == "__main__":
    main()
