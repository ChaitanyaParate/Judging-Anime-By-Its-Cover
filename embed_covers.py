"""
Phase 1: Pre-compute CLIP visual embeddings for all 30k cover images.
Run ONCE to generate cover_embeddings.npy and embedding_index.json.
GPU is enforced — will raise an error if CUDA is not available.
"""

import os
import json
import sqlite3
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

# ── Config ──────────────────────────────────────────────────────────────────
DB_PATH = "anime_data.db"
COVERS_DIR = "covers"
EMBEDDINGS_PATH = "cover_embeddings.npy"
INDEX_PATH = "embedding_index.json"
# Local model path — download once with: python download_model.py
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "clip-vit-base-patch32")
BATCH_SIZE = 256  # images per GPU batch

# ── Device selection (GPU preferred, CPU fallback) ───────────────────────────
if torch.cuda.is_available() and os.environ.get("FORCE_CPU") != "1":
    DEVICE = "cuda"
    print(f"[embed_covers] ✔ GPU detected: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = "cpu"
    if os.environ.get("FORCE_CPU") == "1":
        print("[embed_covers] ⚠ FORCE_CPU=1 set — running on CPU (slow).")
    else:
        print(
            "[embed_covers] ⚠ CUDA not available in this session. Running on CPU.\n"
            "  Tip: For GPU speed, run from a full terminal where 'nvidia-smi' works."
        )
print(f"[embed_covers] Device: {DEVICE}")

# ── Load model ───────────────────────────────────────────────────────────────
if not os.path.isdir(MODEL_PATH):
    raise FileNotFoundError(
        f"Local CLIP model not found at: {MODEL_PATH}\n"
        f"Please run: python download_model.py"
    )
print(f"[embed_covers] Loading CLIP model from: {MODEL_PATH}")
model = CLIPModel.from_pretrained(MODEL_PATH, local_files_only=True).to(DEVICE)
processor = CLIPProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
model.eval()


def load_existing_index():
    """Load existing index so we can resume interrupted runs."""
    if os.path.exists(INDEX_PATH) and os.path.exists(EMBEDDINGS_PATH):
        with open(INDEX_PATH, "r") as f:
            index = json.load(f)  # {str(mal_id): row_idx}
        embeddings = np.load(EMBEDDINGS_PATH)
        print(f"[embed_covers] Resuming — {len(index)} embeddings already computed.")
        return index, list(embeddings)
    return {}, []


def save_index(index, embeddings_list):
    np.save(EMBEDDINGS_PATH, np.array(embeddings_list, dtype=np.float32))
    with open(INDEX_PATH, "w") as f:
        json.dump(index, f)


def embed_batch(image_paths):
    """Embed a batch of image paths → (numpy array [B, 512], valid_indices list)."""
    images = []
    valid_indices = []  # indices into image_paths that loaded successfully
    for i, p in enumerate(image_paths):
        try:
            img = Image.open(p).convert("RGB")
            images.append(img)
            valid_indices.append(i)
        except Exception:
            pass  # skip corrupt images

    if not images:
        return np.array([]), []

    inputs = processor(images=images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(DEVICE)
    with torch.no_grad():
        # In transformers 5.x, get_image_features routes through vision_model
        # Use vision_model directly + project to get the 512-d CLIP embedding
        vision_out = model.vision_model(pixel_values=pixel_values)
        pooled = vision_out.pooler_output  # [B, hidden_size]
        feats = model.visual_projection(pooled)  # [B, 512]
        feats = feats / feats.norm(dim=-1, keepdim=True)  # L2 normalise

    return feats.cpu().numpy(), valid_indices



def main():
    # ── Fetch records from DB ────────────────────────────────────────────────
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT mal_id, local_image_path FROM anime WHERE local_image_path IS NOT NULL")
    records = c.fetchall()
    conn.close()

    print(f"[embed_covers] {len(records)} anime with cover images found in DB.")

    # ── Resume support ───────────────────────────────────────────────────────
    index, embeddings_list = load_existing_index()
    done_ids = set(index.keys())

    pending = [(mid, path) for mid, path in records if str(mid) not in done_ids]
    print(f"[embed_covers] {len(pending)} images left to embed.")

    if not pending:
        print("[embed_covers] Nothing to do — index is complete.")
        return

    # ── Batch embed ──────────────────────────────────────────────────────────
    batch_mal_ids = []
    batch_paths = []

    save_every = 5000  # checkpoint save interval
    total_saved = 0

    for mal_id, local_path in tqdm(pending, desc="Embedding covers"):
        full_path = local_path if os.path.isabs(local_path) else os.path.join(os.getcwd(), local_path)
        if not os.path.exists(full_path):
            continue
        batch_mal_ids.append(mal_id)
        batch_paths.append(full_path)

        if len(batch_paths) >= BATCH_SIZE:
            vecs, valid_indices = embed_batch(batch_paths)
            for embed_i, orig_i in enumerate(valid_indices):
                mid = batch_mal_ids[orig_i]
                index[str(mid)] = len(embeddings_list)
                embeddings_list.append(vecs[embed_i])
            batch_mal_ids, batch_paths = [], []

            # Periodic checkpoint save
            total_saved += BATCH_SIZE
            if total_saved % save_every < BATCH_SIZE:
                save_index(index, embeddings_list)
                print(f"\n[embed_covers] Checkpoint: {len(embeddings_list)} embeddings saved.")


    # ── Flush remaining ──────────────────────────────────────────────────────
    if batch_paths:
        vecs, valid_indices = embed_batch(batch_paths)
        for embed_i, orig_i in enumerate(valid_indices):
            mid = batch_mal_ids[orig_i]
            index[str(mid)] = len(embeddings_list)
            embeddings_list.append(vecs[embed_i])


    # ── Save ─────────────────────────────────────────────────────────────────
    save_index(index, embeddings_list)
    print(f"[embed_covers] Done. Saved {len(embeddings_list)} embeddings → {EMBEDDINGS_PATH}")
    print(f"[embed_covers] Index saved → {INDEX_PATH}")


if __name__ == "__main__":
    main()
