import sqlite3
import cv2
import numpy as np
import colorgram
from facenet_pytorch import MTCNN
import torch
import os
from tqdm import tqdm
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {device}")

mtcnn = MTCNN(keep_all=True, device=device)

def calculate_entropy(histogram):
    histogram = histogram[histogram > 0]
    histogram = histogram / np.sum(histogram)
    return -np.sum(histogram * np.log2(histogram))

def extract_features(image_path):
    features = {}
    
    img = cv2.imread(image_path)
    if img is None:
        return None
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w, _ = img.shape
    total_pixels = h * w
    
    # 1. Faces
    # facenet-pytorch mtcnn expects RGB image (numpy array or PIL image)
    try:
        boxes, probs = mtcnn.detect(img_rgb)
    except Exception as e:
        boxes, probs = None, None

    features['face_count'] = 0
    features['face_area_ratio'] = 0.0
    if boxes is not None:
        valid_boxes = [b for b, p in zip(boxes, probs) if p is not None and p > 0.90]
        features['face_count'] = len(valid_boxes)
        face_area = sum([(b[2]-b[0])*(b[3]-b[1]) for b in valid_boxes])
        features['face_area_ratio'] = min(face_area / total_pixels, 1.0)
        
    # 2. Color Science
    h_channel, s_channel, v_channel = cv2.split(img_hsv)
    features['brightness_mean'] = float(np.mean(v_channel))
    features['brightness_std'] = float(np.std(v_channel))
    features['saturation_mean'] = float(np.mean(s_channel))
    
    # Entropy
    hist_hsv = cv2.calcHist([img_hsv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    features['color_entropy'] = float(calculate_entropy(hist_hsv.flatten()))
    
    # Dominant Colors
    img_small = cv2.resize(img_rgb, (100, 100))
    pil_img = Image.fromarray(img_small)
    colors = colorgram.extract(pil_img, 1)
    if len(colors) > 0:
        c = colors[0].rgb
        features['dominant_r'] = c.r
        features['dominant_g'] = c.g
        features['dominant_b'] = c.b
    else:
        features['dominant_r'] = 0
        features['dominant_g'] = 0
        features['dominant_b'] = 0
        
    # 3. Structure / Edges
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img_gray, 100, 200)
    edge_pixels = np.sum(edges > 0)
    features['edge_density'] = float(edge_pixels / total_pixels)
    
    return features

def setup_features_table(cursor):
    cursor.execute('''CREATE TABLE IF NOT EXISTS anime_features
                      (mal_id INTEGER PRIMARY KEY,
                       face_count INTEGER,
                       face_area_ratio REAL,
                       brightness_mean REAL,
                       brightness_std REAL,
                       saturation_mean REAL,
                       color_entropy REAL,
                       dominant_r INTEGER,
                       dominant_g INTEGER,
                       dominant_b INTEGER,
                       edge_density REAL,
                       FOREIGN KEY(mal_id) REFERENCES anime(mal_id))''')

def main():
    conn = sqlite3.connect("anime_data.db")
    c = conn.cursor()
    setup_features_table(c)
    
    c.execute("SELECT mal_id, local_image_path FROM anime WHERE local_image_path IS NOT NULL")
    records = c.fetchall()
    
    print(f"Found {len(records)} images to process.")
    
    c.execute("SELECT mal_id FROM anime_features")
    processed = set(row[0] for row in c.fetchall())
    
    records_to_process = [r for r in records if r[0] not in processed]
    print(f"{len(processed)} already processed. {len(records_to_process)} remaining.")
    
    batch_size = 100
    updates = []
    
    for mal_id, local_path in tqdm(records_to_process, desc="Extracting Features"):
        if not os.path.exists(local_path):
            continue
            
        feats = extract_features(local_path)
        if feats is not None:
            updates.append((
                mal_id,
                feats['face_count'], feats['face_area_ratio'],
                feats['brightness_mean'], feats['brightness_std'],
                feats['saturation_mean'], feats['color_entropy'],
                feats['dominant_r'], feats['dominant_g'], feats['dominant_b'],
                feats['edge_density']
            ))
            
        if len(updates) >= batch_size:
            c.executemany('''INSERT INTO anime_features VALUES (?,?,?,?,?,?,?,?,?,?,?)''', updates)
            conn.commit()
            updates = []
            
    if updates:
        c.executemany('''INSERT INTO anime_features VALUES (?,?,?,?,?,?,?,?,?,?,?)''', updates)
        conn.commit()
        
    conn.close()
    print("Feature extraction complete!")

if __name__ == "__main__":
    main()
