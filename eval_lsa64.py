import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from model import SignClassifier
from dataset_loader import get_dataloaders
from train_utils import evaluate

def main():
    device = config.DEVICE
    
    # Cargar info de clases
    class_info_path = os.path.join(config.MODELS_DIR, "lsa64_classes.json")
    with open(class_info_path) as f:
        class_info = json.load(f)
    
    idx_to_class = {int(k): v for k, v in class_info["idx_to_class"].items()}
    class_to_idx = class_info["class_to_idx"]
    num_classes = class_info["num_classes"]
    
    print("Cargando modelo...")
    ckpt_path = os.path.join(config.MODELS_DIR, "lsa64_finetuned.pth")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    hidden_dim = ckpt["model_state_dict"]["input_proj.0.weight"].shape[0]
    
    model = SignClassifier(
        input_dim=config.INPUT_DIM,
        hidden_dim=hidden_dim,
        num_layers=config.MODEL_NUM_LSTM_LAYERS,
        num_classes=num_classes,
        dropout=config.MODEL_DROPOUT,
        bidirectional=config.MODEL_BIDIRECTIONAL,
    ).to(device)
    
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    print("Cargando test set...")
    loaders = get_dataloaders(
        "lsa64",
        batch_size=32,
        max_seq_len=config.MAX_SEQ_LEN,
        augment_train=False,
        class_to_idx=class_to_idx
    )
    test_loader = loaders["test"]
    
    criterion = nn.CrossEntropyLoss()
    print("Evaluando...")
    _, test_acc, all_preds, all_labels = evaluate(model, test_loader, criterion, device)
    
    print(f"Accuracy Test: {test_acc:.4f}")
    
    # Calcular matriz de confusión
    cm = confusion_matrix(all_labels, all_preds)
    
    # Encontrar mejores y peores
    class_accs = cm.diagonal() / cm.sum(axis=1)
    
    class_names = [idx_to_class[i] for i in range(num_classes)]
    
    # Mejores 5
    best_idx = np.argsort(class_accs)[::-1][:5]
    print("\n[MEJORES CLASES]")
    for i in best_idx:
        print(f"- {class_names[i]}: {class_accs[i]*100:.1f}%")
        
    # Peores 5
    worst_idx = np.argsort(class_accs)[:5]
    print("\n[PEORES CLASES]")
    for i in worst_idx:
        print(f"- {class_names[i]}: {class_accs[i]*100:.1f}%")
        
    # Confusiones más comunes
    np.fill_diagonal(cm, 0)
    print("\n[CONFUSIONES MÁS COMUNES]")
    confusions = []
    for i in range(num_classes):
        for j in range(num_classes):
            if cm[i, j] > 0:
                confusions.append((cm[i, j], class_names[i], class_names[j]))
    
    confusions.sort(reverse=True)
    for count, true_cls, pred_cls in confusions[:10]:
        print(f"- '{true_cls}' confundido con '{pred_cls}' ({count} veces)")
        
    # Guardar gráfico
    plt.figure(figsize=(24, 20))
    sns.heatmap(confusion_matrix(all_labels, all_preds), cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names, cbar=False)
    plt.title("Matriz de Confusión LSA64")
    plt.xticks(rotation=90)
    plt.tight_layout()
    out_img = os.path.join(config.RESULTS_DIR, "lsa64_confusion_eval.png")
    plt.savefig(out_img, dpi=150)
    print(f"\nMatriz guardada en: {out_img}")

if __name__ == "__main__":
    main()
