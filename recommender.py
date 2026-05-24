"""
Phase 4: Recommendation Engine
Loads precomputed CLIP embeddings and performs:
  1. Embeds the user's liked anime covers (or looks up from index)
  2. Builds user taste vector (mean of liked anime embeddings)
  3. Blends with CLIP text embedding of preference (70% visual, 30% text)
  4. Cosine similarity search over all 30k embeddings
  5. Metadata re-ranking (genre filter + MAL score + popularity)
  6. Returns top-N recommendations
"""

import os
import json
import sqlite3
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import NamedTuple

from input_parser import AnimeEntry

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH = "anime_data.db"
EMBEDDINGS_PATH = "cover_embeddings.npy"
INDEX_PATH = "embedding_index.json"
# Local model path — download once with: python download_model.py
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "clip-vit-large-patch14")

# Blending weights
W_VISUAL = 0.70  # user taste vector (visual similarity)
W_TEXT = 0.30  # preference text embedding

# Final score weights
W_SIM = 0.55
W_SCORE = 0.28
W_POP = 0.17

# Minimum score threshold for candidates
MIN_SCORE = 5.5
MIN_SCORED_BY = 500


# ── Lazy CLIP loader ──────────────────────────────────────────────────────────
_clip_model = None
_clip_processor = None
_device = None


def _get_clip():
    global _clip_model, _clip_processor, _device
    if _clip_model is None:
        if not os.path.isdir(MODEL_PATH):
            raise FileNotFoundError(
                f"Local CLIP model not found at: {MODEL_PATH}\n"
                f"Please run: python download_model.py"
            )
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _clip_model = CLIPModel.from_pretrained(MODEL_PATH, local_files_only=True).to(_device)
        _clip_processor = CLIPProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
        _clip_model.eval()
    return _clip_model, _clip_processor, _device


# ── Embedding index ───────────────────────────────────────────────────────────
_embeddings_matrix: np.ndarray | None = None
_index: dict | None = None
_reverse_index: dict | None = None  # row_idx → mal_id


def _load_index():
    global _embeddings_matrix, _index, _reverse_index
    if _embeddings_matrix is not None:
        return

    if not os.path.exists(EMBEDDINGS_PATH) or not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(
            f"Embedding index not found. Please run:\n"
            f"  python embed_covers.py\n"
            f"to precompute CLIP embeddings for all cover images."
        )

    _embeddings_matrix = np.load(EMBEDDINGS_PATH)  # [N, D] where D=768 for large model
    with open(INDEX_PATH, "r") as f:
        _index = json.load(f)  # {str(mal_id): row_idx}
    _reverse_index = {v: int(k) for k, v in _index.items()}
    print(f"[recommender] Loaded {_embeddings_matrix.shape[0]} embeddings from index.")


# ── DB helpers ────────────────────────────────────────────────────────────────
def _load_anime_db() -> dict[int, dict]:
    """Load all anime metadata from DB."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT mal_id, title, title_english, title_japanese,
               genres, score, scored_by, members, local_image_path
        FROM anime
    """)
    rows = c.fetchall()
    conn.close()

    db = {}
    for row in rows:
        mid, title, en, jp, genres, score, scored_by, members, img = row
        db[mid] = {
            "title": title or "",
            "title_english": en or "",
            "title_japanese": jp or "",
            "genres": genres or "",
            "score": score or 0.0,
            "scored_by": scored_by or 0,
            "members": members or 0,
            "local_image_path": img,
        }
    return db


# ── Image embedding ───────────────────────────────────────────────────────────
def _embed_image(image_path: str) -> np.ndarray | None:
    """Embed a single image file via CLIP → 512-d L2-normalised vector."""
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"  [recommender] Could not open image {image_path}: {e}")
        return None

    model, processor, device = _get_clip()
    inputs = processor(images=[img], return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    with torch.no_grad():
        vision_out = model.vision_model(pixel_values=pixel_values)
        pooled = vision_out.pooler_output
        feat = model.visual_projection(pooled)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy()[0]



def _get_embedding_for_entry(entry: AnimeEntry) -> np.ndarray | None:
    """Get CLIP embedding for a liked anime — from index if available, else embed live."""
    _load_index()
    key = str(entry.mal_id)
    if key in _index:
        row_idx = _index[key]
        return _embeddings_matrix[row_idx]

    # Not in index: embed the local image live
    if entry.local_image_path and os.path.exists(entry.local_image_path):
        print(f"  [recommender] Live-embedding {entry.title} (not in precomputed index)")
        return _embed_image(entry.local_image_path)

    print(f"  [recommender] WARNING: No embedding available for {entry.title}")
    return None


# ── Cosine similarity ─────────────────────────────────────────────────────────
def _cosine_sim(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Compute cosine similarity of query [512] against matrix [N, 512].
    Assumes both are already L2-normalised."""
    return matrix @ query  # dot product == cosine sim when normalised


# ── Result dataclass ──────────────────────────────────────────────────────────
class Recommendation(NamedTuple):
    rank: int
    mal_id: int
    title: str
    title_english: str
    title_japanese: str
    genres: str
    score: float
    members: int
    local_image_path: str | None
    similarity: float
    final_score: float


# ── Main recommender function ─────────────────────────────────────────────────
def recommend(
    liked_anime: list[AnimeEntry],
    preference_text_embed: np.ndarray | None = None,
    genre_filter: set[str] | None = None,
    top_n: int = 6,
    n_candidates: int = 150,
) -> list[Recommendation]:
    """
    Produce top-N anime recommendations.

    Args:
        liked_anime: List of AnimeEntry objects from input_parser
        preference_text_embed: 512-d CLIP text embedding of preference string
        genre_filter: Set of MAL genre names to soft-filter by
        top_n: Number of final recommendations to return
        n_candidates: Number of visual candidates to retrieve before re-ranking
    """
    _load_index()
    anime_db = _load_anime_db()

    # ── 1. Build liked anime embeddings ───────────────────────────────────────
    liked_vecs = []
    for entry in liked_anime:
        vec = _get_embedding_for_entry(entry)
        if vec is not None:
            liked_vecs.append(vec)

    if not liked_vecs:
        raise ValueError("Could not obtain embeddings for any liked anime. "
                         "Check that cover images exist and embed_covers.py has been run.")

    # ── 2. User taste vector (weighted by Worth Level if available) ───────────
    weights = np.array([entry.weight for entry in liked_anime
                        if _get_embedding_for_entry(entry) is not None], dtype=np.float32)
    # Re-collect only entries that produced valid embeddings
    valid_entries = [entry for entry in liked_anime
                    if _get_embedding_for_entry(entry) is not None]
    valid_weights = np.array([e.weight for e in valid_entries], dtype=np.float32)

    if valid_weights.sum() == 0:
        valid_weights = np.ones(len(liked_vecs), dtype=np.float32)

    if len(set(valid_weights.tolist())) == 1:
        # All weights equal — use simple mean (faster)
        taste_vec = np.mean(liked_vecs, axis=0)
    else:
        # Weighted mean: High-worth anime dominate the query vector
        taste_vec = np.average(liked_vecs, axis=0, weights=valid_weights)
        n_high = int((valid_weights > 1.0).sum())
        n_low  = int((valid_weights < 1.0).sum())
        print(f"  [recommender] Weighted query: {n_high} High, "
              f"{len(liked_vecs)-n_high-n_low} Medium, {n_low} Low entries")
    taste_vec = taste_vec / (np.linalg.norm(taste_vec) + 1e-9)  # re-normalise

    # ── 3. Blend with preference text embedding ───────────────────────────────
    if preference_text_embed is not None:
        query_vec = W_VISUAL * taste_vec + W_TEXT * preference_text_embed
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    else:
        query_vec = taste_vec

    # ── 4. Cosine similarity over full index ──────────────────────────────────
    sims = _cosine_sim(query_vec, _embeddings_matrix)  # [N]

    # ── 5. Get top candidates, excluding liked anime ──────────────────────────
    liked_ids = {entry.mal_id for entry in liked_anime}
    liked_row_idxs = {_index[str(mid)] for mid in liked_ids if str(mid) in _index}

    # Mask out liked anime
    masked_sims = sims.copy()
    for row_idx in liked_row_idxs:
        masked_sims[row_idx] = -999.0

    top_candidate_idxs = np.argsort(masked_sims)[::-1][:n_candidates]

    # ── 6. Normalise MAL score and members for re-ranking ─────────────────────
    all_scores = np.array([anime_db[mid]["score"] for mid in anime_db if anime_db[mid]["score"] > 0])
    score_min, score_max = all_scores.min(), all_scores.max()
    score_range = score_max - score_min + 1e-9

    all_members = np.array([anime_db[mid]["members"] for mid in anime_db])
    members_max = all_members.max() + 1e-9

    # ── 7. Re-rank with metadata ──────────────────────────────────────────────
    candidates = []
    for row_idx in top_candidate_idxs:
        mal_id = _reverse_index.get(row_idx)
        if mal_id is None or mal_id not in anime_db:
            continue

        info = anime_db[mal_id]

        # Hard filters
        if info["score"] < MIN_SCORE:
            continue
        if info["scored_by"] < MIN_SCORED_BY:
            continue

        # Genre soft-boost
        genre_boost = 0.0
        if genre_filter and info["genres"]:
            anime_genres = {g.strip() for g in info["genres"].split(",")}
            overlap = len(anime_genres & genre_filter)
            genre_boost = min(overlap * 0.0625, 0.25)  # max +0.25 boost

        # Normalised sub-scores
        norm_score = (info["score"] - score_min) / score_range
        norm_pop = info["members"] / members_max
        vis_sim = float(masked_sims[row_idx])

        final = W_SIM * vis_sim + W_SCORE * norm_score + W_POP * norm_pop + genre_boost

        candidates.append((final, vis_sim, mal_id, info))

    # Sort by final score descending
    candidates.sort(key=lambda x: x[0], reverse=True)

    # ── 8. Build results ──────────────────────────────────────────────────────
    results = []
    for rank, (final, vis_sim, mal_id, info) in enumerate(candidates[:top_n], start=1):
        results.append(Recommendation(
            rank=rank,
            mal_id=mal_id,
            title=info["title"],
            title_english=info["title_english"],
            title_japanese=info["title_japanese"],
            genres=info["genres"],
            score=info["score"],
            members=info["members"],
            local_image_path=info["local_image_path"],
            similarity=round(vis_sim, 4),
            final_score=round(final, 4),
        ))

    return results
