"""
08_export_model.py — Exporta el modelo final a ONNX y TorchScript.

Uso:
    python 08_export_model.py [--model-path models/lsp_final.pth] [--dataset lsp_pucp]
"""
import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from model import SignClassifier
from utils import load_checkpoint

import torch


def export_onnx(model, export_path, input_dim, max_seq_len):
    """Exporta modelo a formato ONNX."""
    model.eval()

    # Dummy input
    dummy_seq = torch.randn(1, max_seq_len, input_dim)
    dummy_len = torch.tensor([max_seq_len], dtype=torch.long)

    # ONNX no soporta bien pack_padded_sequence, así que usamos modo sin lengths
    # Crear wrapper que no use lengths
    class ModelWrapper(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model

        def forward(self, x):
            return self.base_model(x, lengths=None)

    wrapper = ModelWrapper(model)
    wrapper.eval()

    os.makedirs(os.path.dirname(export_path), exist_ok=True)

    torch.onnx.export(
        wrapper,
        dummy_seq,
        export_path,
        input_names=["keypoint_sequence"],
        output_names=["class_logits"],
        dynamic_axes={
            "keypoint_sequence": {0: "batch_size", 1: "sequence_length"},
            "class_logits": {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    print(f"[EXPORT] ONNX exportado: {export_path}")
    print(f"  Tamaño: {os.path.getsize(export_path) / 1024 / 1024:.2f} MB")
    return export_path


def export_torchscript(model, export_path, input_dim, max_seq_len):
    """Exporta modelo a TorchScript via tracing."""
    model.eval()

    class ModelWrapper(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model

        def forward(self, x):
            return self.base_model(x, lengths=None)

    wrapper = ModelWrapper(model)
    wrapper.eval()

    dummy_seq = torch.randn(1, max_seq_len, input_dim)

    os.makedirs(os.path.dirname(export_path), exist_ok=True)

    traced = torch.jit.trace(wrapper, dummy_seq)
    traced.save(export_path)

    print(f"[EXPORT] TorchScript exportado: {export_path}")
    print(f"  Tamaño: {os.path.getsize(export_path) / 1024 / 1024:.2f} MB")
    return export_path


def validate_onnx(onnx_path, model, input_dim, max_seq_len):
    """Valida que la salida ONNX coincida con PyTorch."""
    try:
        import onnx
        import onnxruntime as ort

        # Verificar modelo ONNX
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("[EXPORT] ✓ Modelo ONNX válido")

        # Comparar salidas
        model.eval()
        dummy_seq = torch.randn(1, max_seq_len, input_dim)

        class ModelWrapper(torch.nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model
            def forward(self, x):
                return self.base_model(x, lengths=None)

        wrapper = ModelWrapper(model)
        wrapper.eval()

        with torch.no_grad():
            pytorch_out = wrapper(dummy_seq).numpy()

        sess = ort.InferenceSession(onnx_path)
        onnx_out = sess.run(None, {"keypoint_sequence": dummy_seq.numpy()})[0]

        max_diff = np.abs(pytorch_out - onnx_out).max()
        print(f"[EXPORT] Diferencia máxima PyTorch vs ONNX: {max_diff:.2e}")

        if max_diff < 1e-4:
            print("[EXPORT] ✓ Validación exitosa (diff < 1e-4)")
        else:
            print("[EXPORT] ⚠ Diferencia mayor a 1e-4, revisar")

    except ImportError:
        print("[EXPORT] ⚠ onnxruntime no instalado, saltando validación")
    except Exception as e:
        print(f"[EXPORT] ⚠ Error en validación: {e}")


def main():
    parser = argparse.ArgumentParser(description="Exporta modelo a ONNX/TorchScript")
    parser.add_argument("--model-path", type=str,
                        default=config.LSP_TRAIN["checkpoint"])
    parser.add_argument("--dataset", type=str, default="lsp_pucp",
                        choices=["wlasl", "autsl", "lsp_pucp"])
    parser.add_argument("--no-onnx", action="store_true")
    parser.add_argument("--no-torchscript", action="store_true")
    args = parser.parse_args()

    # Obtener número de clases
    class_files = {
        "wlasl": os.path.join(config.MODELS_DIR, "wlasl_classes.json"),
        "autsl": os.path.join(config.MODELS_DIR, "autsl_classes.json"),
        "lsp_pucp": os.path.join(config.MODELS_DIR, "lsp_classes.json"),
    }

    class_info_path = class_files[args.dataset]
    if not os.path.exists(class_info_path):
        print(f"ERROR: No se encontró {class_info_path}")
        print("Ejecuta primero el script de entrenamiento correspondiente.")
        sys.exit(1)

    with open(class_info_path) as f:
        class_info = json.load(f)
    num_classes = class_info["num_classes"]

    # Crear modelo
    model = SignClassifier(
        input_dim=config.INPUT_DIM,
        hidden_dim=config.MODEL_HIDDEN_DIM,
        num_layers=config.MODEL_NUM_LSTM_LAYERS,
        num_classes=num_classes,
        dropout=0.0,  # Sin dropout para inferencia
        bidirectional=config.MODEL_BIDIRECTIONAL,
    )

    # Cargar pesos
    if not os.path.exists(args.model_path):
        print(f"ERROR: Checkpoint no encontrado: {args.model_path}")
        sys.exit(1)

    load_checkpoint(model, args.model_path, device="cpu")
    model.eval()

    print(f"\nModelo: {args.model_path}")
    print(f"Clases: {num_classes}")
    print(f"Input dim: {config.INPUT_DIM}")
    total, trainable = model.count_parameters()
    print(f"Parámetros: {total:,}")

    # Exportar ONNX
    if not args.no_onnx:
        onnx_path = os.path.join(
            config.EXPORT_DIR,
            f"sign_classifier_{args.dataset}.onnx"
        )
        export_onnx(model, onnx_path, config.INPUT_DIM, config.MAX_SEQ_LEN)
        validate_onnx(onnx_path, model, config.INPUT_DIM, config.MAX_SEQ_LEN)

    # Exportar TorchScript
    if not args.no_torchscript:
        ts_path = os.path.join(
            config.EXPORT_DIR,
            f"sign_classifier_{args.dataset}.pt"
        )
        export_torchscript(model, ts_path, config.INPUT_DIM, config.MAX_SEQ_LEN)

    # Guardar metadata de exportación
    export_meta = {
        "dataset": args.dataset,
        "num_classes": num_classes,
        "input_dim": config.INPUT_DIM,
        "max_seq_len": config.MAX_SEQ_LEN,
        "num_keypoints": config.NUM_KEYPOINTS,
        "class_to_idx": class_info.get("class_to_idx", {}),
        "model_architecture": "SignClassifier (BiLSTM + Attention)",
        "hidden_dim": config.MODEL_HIDDEN_DIM,
        "num_lstm_layers": config.MODEL_NUM_LSTM_LAYERS,
    }

    meta_path = os.path.join(config.EXPORT_DIR,
                             f"sign_classifier_{args.dataset}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(export_meta, f, ensure_ascii=False, indent=2)

    print(f"\nMetadata: {meta_path}")
    print("✓ Exportación completada.")


if __name__ == "__main__":
    main()
