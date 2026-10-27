import cv2
import numpy as np
import os
import sys

# Add current path to sys so config and utils are found
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import get_mediapipe_hands, detect_hand_keypoints
import config

C_YELLOW = (81, 196, 253)[::-1]
C_ACCENT = (223, 111, 168)[::-1]

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

def main():
    video_path = "data/raw/autsl/AUTSL/train/signer0_sample1_color.mp4"
    out_path = os.path.join(config.RESULTS_DIR, "debug_landmarks.mp4")
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    hands = get_mediapipe_hands()
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    
    print(f"Generando video en {out_path}...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        kps = detect_hand_keypoints(frame_rgb, hands)
        
        draw_landmarks_custom(frame, kps, C_ACCENT)
        out.write(frame)
        
    cap.release()
    out.release()
    print("¡Video generado!")

if __name__ == "__main__":
    main()
