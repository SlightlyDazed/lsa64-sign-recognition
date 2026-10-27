"""
10_live_inference.py — Inferencia en vivo (Aesthetic: monitor_emg_gesture)
"""
import cv2
import numpy as np
import torch
import json
import os
import sys
import time
import collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from model import SignClassifier
from utils import get_mediapipe_hands, extract_keypoints_frame, normalize_keypoints

# Colores estéticos de monitor_emg_gesture
BG_DARK  = (20, 22, 23)
BG_SURF  = (25, 27, 28)
TEXT_C   = (200, 204, 205)
DIVIDER  = (54, 56, 57)
C_GREEN  = (109, 170, 69)[::-1]
C_RED    = (110, 110, 253)[::-1]
C_YELLOW = (81, 196, 253)[::-1]
C_ACCENT = (223, 111, 168)[::-1]
C_CYAN   = (163, 152, 79)[::-1]
FONT = cv2.FONT_HERSHEY_SIMPLEX

def draw_landmarks_custom(frame, kps, color):
    # kps es (42, 3) -> 21 left, 21 right
    connections = [
        (0,1), (1,2), (2,3), (3,4),       # thumb
        (0,5), (5,6), (6,7), (7,8),       # index
        (5,9), (9,10), (10,11), (11,12),  # middle
        (9,13), (13,14), (14,15), (15,16),# ring
        (13,17), (0,17), (17,18), (18,19), (19,20) # pinky
    ]
    for offset in [0, 21]:
        # Revisar si hay mano detectada (conf > 0 o no ceros)
        hand_pts = kps[offset:offset+21]
        if np.all(hand_pts == 0): continue
        
        # Connections
        for i, j in connections:
            pt1 = (int(hand_pts[i][0]), int(hand_pts[i][1]))
            pt2 = (int(hand_pts[j][0]), int(hand_pts[j][1]))
            if pt1 != (0,0) and pt2 != (0,0):
                cv2.line(frame, pt1, pt2, (255,255,255), 2, cv2.LINE_AA)
        
        # Nodes
        for i, pt in enumerate(hand_pts):
            x, y = int(pt[0]), int(pt[1])
            if (x, y) != (0,0):
                c = C_YELLOW if i % 4 == 0 else color
                cv2.circle(frame, (x, y), 5, c, -1, cv2.LINE_AA)
                cv2.circle(frame, (x, y), 5, (255,255,255), 1, cv2.LINE_AA)

def draw_panel(frame, h, w, pred_class, conf_val, active, fps):
    panel_w = 300
    panel = np.full((h, panel_w, 3), BG_SURF, dtype=np.uint8)
    cv2.line(panel, (0,0), (0,h), DIVIDER, 2)
    
    cv2.putText(panel, "LSA64 RECOGNITION", (20, 40), FONT, 0.7, TEXT_C, 2, cv2.LINE_AA)
    cv2.putText(panel, "BiLSTM Sign Language", (20, 65), FONT, 0.4, C_ACCENT, 1, cv2.LINE_AA)
    cv2.line(panel, (20, 80), (panel_w-20, 80), DIVIDER, 1)
    
    cv2.putText(panel, "ESTADO DEL MODELO", (20, 110), FONT, 0.4, TEXT_C, 1, cv2.LINE_AA)
    status_color = C_GREEN if active else C_RED
    status_text = "ACTIVO" if active else "ESPERANDO"
    cv2.putText(panel, status_text, (20, 140), FONT, 0.8, status_color, 2, cv2.LINE_AA)
    
    cv2.putText(panel, "PREDICCION", (20, 200), FONT, 0.4, TEXT_C, 1, cv2.LINE_AA)
    color_pred = C_GREEN if conf_val > 0.5 else C_YELLOW
    cv2.putText(panel, pred_class.upper(), (20, 240), FONT, 1.2, color_pred, 3, cv2.LINE_AA)
    
    cv2.putText(panel, "CONFIANZA", (20, 300), FONT, 0.4, TEXT_C, 1, cv2.LINE_AA)
    cv2.rectangle(panel, (20, 320), (20+int(200*conf_val), 340), color_pred, -1)
    cv2.rectangle(panel, (20, 320), (220, 340), DIVIDER, 1)
    cv2.putText(panel, f"{conf_val*100:.1f}%", (230, 335), FONT, 0.5, TEXT_C, 1, cv2.LINE_AA)
    
    cv2.putText(panel, f"FPS: {fps:.1f}", (20, h-30), FONT, 0.4, TEXT_C, 1, cv2.LINE_AA)
    
    out = np.hstack([frame, panel])
    return out

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="models/lsa64_finetuned.pth")
    parser.add_argument("--dataset", type=str, default="lsa64")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    device = config.DEVICE

    class_info_path = os.path.join(config.MODELS_DIR, f"{args.dataset}_classes.json")
    if not os.path.exists(class_info_path):
        print(f"No se encontró el mapeo de clases en {class_info_path}")
        return
        
    with open(class_info_path) as f:
        class_info = json.load(f)
        
    idx_to_class = {int(k): v for k, v in class_info["idx_to_class"].items()}
    num_classes = len(idx_to_class)

    print(f"Cargando modelo desde {args.model_path}...")
    try:
        ckpt_tmp = torch.load(args.model_path, map_location="cpu")
        hidden_dim_ckpt = ckpt_tmp["model_state_dict"]["input_proj.0.weight"].shape[0]
        
        model = SignClassifier(
            input_dim=config.INPUT_DIM,
            hidden_dim=hidden_dim_ckpt,
            num_layers=config.MODEL_NUM_LSTM_LAYERS,
            num_classes=num_classes,
            dropout=config.MODEL_DROPOUT,
            bidirectional=config.MODEL_BIDIRECTIONAL,
        ).to(device)
        
        model.load_state_dict(ckpt_tmp["model_state_dict"])
        model.eval()
    except Exception as e:
        print(f"Advertencia: no se pudo cargar el modelo BiLSTM: {e}")
        model = None

    print("Inicializando MediaPipe Hands...")
    hands_model = get_mediapipe_hands()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"No se pudo abrir la cámara {args.camera}.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    sequence = []
    window_size = config.MAX_SEQ_LEN
    fps_buf = collections.deque(maxlen=30)
    
    cv2.namedWindow("Inferencia en vivo (LSA64)", cv2.WINDOW_NORMAL)
    
    pred_class = "---"
    conf_val = 0.0
    
    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret: break
            
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        kps_raw = extract_keypoints_frame(frame, None, use_hands=True)
        active = False
        
        if kps_raw is not None and not np.all(kps_raw == 0):
            draw_landmarks_custom(frame, kps_raw, C_ACCENT)

            # Posición global de la muñeca: usar coords normalizadas MediaPipe (índice 0 de cada mano)
            # kps_raw[:,0] = x_px, kps_raw[:,1] = y_px → convertir a relativo al frame
            lw_x = (kps_raw[0, 0] / w) - 0.5 if kps_raw[0, 0] != 0 else 0.0
            lw_y = (kps_raw[0, 1] / h) - 0.5 if kps_raw[0, 1] != 0 else 0.0
            rw_x = (kps_raw[21, 0] / w) - 0.5 if kps_raw[21, 0] != 0 else 0.0
            rw_y = (kps_raw[21, 1] / h) - 0.5 if kps_raw[21, 1] != 0 else 0.0
            global_pos = np.array([lw_x, lw_y, rw_x, rw_y], dtype=np.float32)

            sequence.append((kps_raw, global_pos))
            active = True
            
            if len(sequence) > window_size:
                sequence.pop(0)
                
            if len(sequence) >= config.MIN_SEQ_LEN and model is not None:
                # Normalizar kps y concatenar posición global
                kps_arr    = np.stack([s[0] for s in sequence])       # (T, 42, 3)
                gpos_arr   = np.stack([s[1] for s in sequence])       # (T, 4)
                kps_normed = normalize_keypoints(kps_arr)              # (T, 42, 3)
                kps_flat   = kps_normed.reshape(len(sequence), -1)    # (T, 126)
                seq_arr    = np.concatenate([kps_flat, gpos_arr], axis=1)  # (T, 130)

                seq_tensor = torch.zeros((1, window_size, config.INPUT_DIM))
                seq_len = len(seq_arr)
                seq_tensor[0, :seq_len, :] = torch.tensor(seq_arr, dtype=torch.float32)
                lengths = torch.tensor([seq_len], dtype=torch.long)
                
                with torch.no_grad():
                    logits = model(seq_tensor.to(device), lengths.to(device))
                    probs = torch.softmax(logits, dim=1)
                    conf, pred_idx = probs.max(1)
                    pred_class = idx_to_class[pred_idx.item()]
                    conf_val = conf.item()
        else:
            # Decay conf_val if hand leaves screen
            conf_val = max(0, conf_val - 0.05)
            if conf_val < 0.1: pred_class = "---"
            sequence.clear()
            
        # Traducción de etiquetas LSA64 a Español
        LSA64_ES_TRANSLATION = {
            "opaque": "opaco", "red": "rojo", "green": "verde", "yellow": "amarillo",
            "bright": "brillante", "light-blue": "celeste", "colors": "colores", "pink": "rosa",
            "women": "mujer", "enemy": "enemigo", "son": "hijo", "man": "hombre",
            "away": "lejos", "drawer": "cajon", "born": "nacer", "learn": "aprender",
            "call": "llamar", "skimmer": "espumadera", "bitter": "amargo", "sweet-milk": "dulce de leche",
            "milk": "leche", "water": "agua", "food": "comida", "argentina": "argentina",
            "uruguay": "uruguay", "country": "pais", "last-name": "apellido", "where": "donde",
            "mock": "burla", "birthday": "cumpleanos", "breakfast": "desayuno", "photo": "foto",
            "hungry": "hambre", "map": "mapa", "coin": "moneda", "music": "musica",
            "ship": "barco", "none": "ninguno", "name": "nombre", "patience": "paciencia",
            "perfume": "perfume", "deaf": "sordo", "trap": "trampa", "rice": "arroz",
            "barbecue": "asado", "candy": "caramelo", "chewing-gum": "chicle", "spaghetti": "fideos",
            "yogurt": "yogur", "accept": "aceptar", "thanks": "gracias", "shut-down": "apagar",
            "appear": "aparecer", "to-land": "aterrizar", "catch": "atrapar", "help": "ayuda",
            "dance": "bailar", "bathe": "banar", "buy": "comprar", "copy": "copiar",
            "run": "correr", "realize": "darse cuenta", "give": "dar", "find": "encontrar"
        }

        pred_class_es = LSA64_ES_TRANSLATION.get(pred_class, pred_class) if pred_class != "---" else "---"

        # --- Voice Feedback ---
        if conf_val > 0.5 and pred_class_es != "---":
            # Usar una variable persistente adjunta a la función para no usar globales
            if not hasattr(main, "last_spoken"):
                main.last_spoken = None
                main.speak_time = 0
                
            if pred_class_es != main.last_spoken or (time.time() - main.speak_time) > 3.0:
                text_to_say = pred_class_es
                
                # Función que correrá en un hilo separado para no trabar la cámara
                def speak_worker(text):
                    try:
                        from gtts import gTTS
                        import subprocess
                        import uuid
                        # Crear archivo temporal único para evitar colisiones
                        mp3_path = f"/tmp/senas_tts_{uuid.uuid4().hex[:6]}.mp3"
                        tts = gTTS(text=text, lang='es')  # <-- Cambiado a español
                        tts.save(mp3_path)
                        # Reproducir con vlc sin interfaz (cvlc)
                        subprocess.run(['cvlc', '--play-and-exit', '--quiet', mp3_path], 
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        os.remove(mp3_path)
                    except Exception as e:
                        print(f"Error TTS: {e}")

                import threading
                threading.Thread(target=speak_worker, args=(text_to_say,), daemon=True).start()
                
                main.last_spoken = pred_class_es
                main.speak_time = time.time()
                
        t1 = time.time()
        fps_buf.append(1.0 / max(t1 - t0, 1e-6))
        
        out_frame = draw_panel(frame, h, w, pred_class_es, conf_val, active, np.mean(fps_buf))
        cv2.imshow("Inferencia en vivo (LSA64)", out_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
