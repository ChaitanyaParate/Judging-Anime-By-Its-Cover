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
import math
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
GENRE_CORR_PATH = "genre_correlation.json"  # cached co-occurrence table
# Local model path — download once with: python download_model.py
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "clip-vit-large-patch14")

# Blending weights
W_VISUAL = 0.70  # user taste vector (visual similarity)
W_TEXT = 0.30  # preference text embedding

# Final score weights  (must sum to 1.0)
W_SIM = 0.45    # visual cover similarity
W_SCORE = 0.28  # MAL community score
W_POP = 0.17    # MAL member count (popularity)
W_GENRE = 0.25  # genre relevance (0 when no genre filter; 0→1 based on overlap fraction)


# Minimum score threshold for candidates
MIN_SCORE = 5.5
MIN_SCORED_BY = 500

# MMR diversity weight (0 = pure relevance, 1 = pure diversity)
# 0.40 means: 60% relevance + 40% penalise similarity to already-picked results
MMR_LAMBDA = 0.40

# Genre relevance threshold: anime scoring below this after normalization
# get a hard-zero genre contribution (filters out weakly-correlated anime).
# 0.13 cuts off pure Drama/Romance anime (Toradora! scores ~0.12 for Psychological)
# while keeping Action/Supernatural anime with genuine thematic alignment.
GENRE_THRESHOLD = 0.13

# Demographic genres need REVERSED correlation direction.
# For Josei/Seinen/Shounen/Shoujo/Kids the table has P(G | Demographic), but
# we need P(Demographic | G) so that Drama anime don't score high for "Josei".
DEMOGRAPHIC_GENRES = {"Josei", "Seinen", "Shoujo", "Shounen", "Kids"}



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


# ── Embedding index & genre correlation ─────────────────────────────────────
_embeddings_matrix: np.ndarray | None = None
_index: dict | None = None
_reverse_index: dict | None = None  # row_idx → mal_id
_genre_corr: dict | None = None     # (anime_genre, requested_genre) → float 0-1


def _build_genre_correlation() -> dict:
    """Build a soft genre correlation table from co-occurrence in the anime DB.

    Returns corr[(G, F)] = P(G | F) = fraction of anime tagged F that are also
    tagged G.  Self-correlation is always 1.0.  Pairs with < MIN_CO occurrences
    are treated as 0.0 to avoid noise from rare genres.

    Usage in scoring:
        genre_relevance = mean over F in genre_filter of:
            max over G in anime_genres of corr.get((G, F), 0.0)
    """
    import sqlite3 as _sq
    from collections import defaultdict

    MIN_CO = 5  # minimum co-occurrences to record a correlation

    conn = _sq.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT genres FROM anime WHERE genres IS NOT NULL AND genres != ''")
    rows = c.fetchall()
    conn.close()

    genre_count: dict[str, int] = defaultdict(int)
    co_count: dict[tuple[str, str], int] = defaultdict(int)

    for (genres_str,) in rows:
        tags = [g.strip() for g in genres_str.split(",") if g.strip()]
        for t in tags:
            genre_count[t] += 1
        for g1 in tags:
            for g2 in tags:  # g2 = the "requested" genre, g1 = anime's genre
                co_count[(g1, g2)] += 1

    corr: dict[tuple[str, str], float] = {}
    for (g1, g2), count in co_count.items():
        if count >= MIN_CO and genre_count[g2] > 0:
            corr[(g1, g2)] = count / genre_count[g2]  # P(g1 | g2), max 1.0 for self

    print(f"[recommender] Genre correlation table: {len(genre_count)} genres, "
          f"{len(corr)} correlated pairs (min_co={MIN_CO})")

    # Persist to disk: JSON with "G1|G2" keys so it's human-readable
    serialisable = {f"{g1}|{g2}": v for (g1, g2), v in corr.items()}
    with open(GENRE_CORR_PATH, "w") as f:
        json.dump(serialisable, f, indent=2, sort_keys=True)
    print(f"[recommender] Saved genre correlation → {GENRE_CORR_PATH}")

    return corr


def _load_genre_corr() -> dict:
    """Load genre correlation from cache file if fresh, else rebuild from DB."""
    db_mtime = os.path.getmtime(DB_PATH) if os.path.exists(DB_PATH) else 0
    cache_mtime = os.path.getmtime(GENRE_CORR_PATH) if os.path.exists(GENRE_CORR_PATH) else 0

    if cache_mtime >= db_mtime and os.path.exists(GENRE_CORR_PATH):
        with open(GENRE_CORR_PATH, "r") as f:
            raw = json.load(f)
        corr = {tuple(k.split("|", 1)): v for k, v in raw.items()}
        pairs = len(corr)
        genres = len({g for g, _ in corr})
        print(f"[recommender] Loaded genre correlation from cache: {genres} genres, {pairs} pairs")
        return corr

    # Cache missing or DB updated — rebuild
    return _build_genre_correlation()


def _load_index():
    global _embeddings_matrix, _index, _reverse_index, _genre_corr
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
    _genre_corr = _load_genre_corr()  # load from cache or rebuild
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

    # ── 5b. Genre-seeded candidate injection ──────────────────────────────────
    # When a genre filter is active, actual genre-tagged anime may not appear in
    # the top-N visual candidates (e.g., Josei anime look different from Shounen).
    # Inject ALL passing anime that have the exact genre tag into the pool so they
    # get a fair chance at ranking via their genre_relevance score.
    injected_idxs: set[int] = set()
    if genre_filter and anime_db:
        # Build set of row_idxs already in the visual candidate pool
        existing_pool = set(top_candidate_idxs.tolist())
        for mid, info in anime_db.items():
            if mid in liked_ids:
                continue
            if info["score"] < MIN_SCORE or info["scored_by"] < MIN_SCORED_BY:
                continue
            if not info["genres"]:
                continue
            anime_tags = {g.strip() for g in info["genres"].split(",") if g.strip()}
            if anime_tags & genre_filter:  # exact tag overlap
                row_idx = _index.get(str(mid))
                if row_idx is not None and row_idx not in existing_pool:
                    injected_idxs.add(row_idx)

    # Merge: visual candidates + genre injections (order: visual first)
    all_candidate_idxs = list(top_candidate_idxs) + list(injected_idxs)

    # ── 6. Normalise MAL score and members for re-ranking ─────────────────────
    all_scores = np.array([anime_db[mid]["score"] for mid in anime_db if anime_db[mid]["score"] > 0])
    score_min, score_max = all_scores.min(), all_scores.max()
    score_range = score_max - score_min + 1e-9

    all_members = np.array([anime_db[mid]["members"] for mid in anime_db])
    members_max = all_members.max() + 1e-9

    # ── 7. Re-rank with metadata ──────────────────────────────────────────────
    candidates = []
    for row_idx in all_candidate_idxs:
        mal_id = _reverse_index.get(row_idx)
        if mal_id is None or mal_id not in anime_db:
            continue

        info = anime_db[mal_id]

        # Hard filters
        if info["score"] < MIN_SCORE:
            continue
        if info["scored_by"] < MIN_SCORED_BY:
            continue

        # Genre relevance: sum all genre correlations, normalize, apply threshold
        #
        # Two correlation directions depending on whether the requested genre is a
        # demographic (Josei, Seinen, Shounen, Shoujo, Kids):
        #
        #  Regular genres  → P(anime_genre | req_genre)
        #    "If I want Horror, how likely is this anime's genre to co-occur?"
        #    e.g. corr[(Action, Horror)] = 0.285
        #
        #  Demographic genres → P(req_genre | anime_genre)   [REVERSED]
        #    "Given this anime has Drama, how likely is it to be Josei?"
        #    e.g. corr[(Josei, Drama)] = 0.013 → Toradora! scores ~0.036 → zeroed
        #    Actual Josei anime: corr[(Josei, Josei)] = 1.0 → strong boost
        genre_relevance = 0.0
        if genre_filter and info["genres"] and _genre_corr is not None:
            anime_genres = [g.strip() for g in info["genres"].split(",") if g.strip()]
            if anime_genres:
                per_req: list[float] = []
                for req_genre in genre_filter:
                    if req_genre in DEMOGRAPHIC_GENRES:
                        # Reversed: P(req_genre | anime_genre)
                        total_corr = sum(
                            _genre_corr.get((req_genre, ag), 0.0) for ag in anime_genres
                        )
                    else:
                        # Standard: P(anime_genre | req_genre)
                        total_corr = sum(
                            _genre_corr.get((ag, req_genre), 0.0) for ag in anime_genres
                        )
                    # Normalize: divide by genre count (max possible = 1.0 per genre)
                    per_req.append(total_corr / len(anime_genres))
                # Multi-genre aggregation: geometric mean.
                # Penalises imbalanced scores more than arithmetic mean but less than min.
                # Examples for "Romance Horror":
                #   Toradora! (0.50, 0.02) → geomean = 0.10 < 0.13 → zeroed ✓
                #   Bakemonogatari (0.29, 0.19) → geomean = 0.23 → passes ✓
                # Examples for "Isekai Horror":
                #   Overlord (0.70, 0.10) → geomean = 0.26 → passes ✓  (min would zero it)
                #   Toradora! (0.05, 0.04) → geomean = 0.045 → zeroed ✓
                if len(per_req) > 1:
                    log_sum = sum(math.log(max(s, 1e-9)) for s in per_req)
                    raw = math.exp(log_sum / len(per_req))
                else:
                    raw = per_req[0]
                genre_relevance = raw if raw >= GENRE_THRESHOLD else 0.0

        # Normalised sub-scores
        norm_score = (info["score"] - score_min) / score_range
        norm_pop = info["members"] / members_max
        vis_sim = float(masked_sims[row_idx])

        final = W_SIM * vis_sim + W_SCORE * norm_score + W_POP * norm_pop + W_GENRE * genre_relevance

        candidates.append((final, vis_sim, mal_id, info))

    # Sort by final score descending
    candidates.sort(key=lambda x: x[0], reverse=True)

    # ── 8. MMR diversity re-ranking ───────────────────────────────────────────
    # Pre-seed with the query vector so even the first pick is penalised for
    # being too close to the taste centroid (prevents Toradora!/Charlotte bias).
    selected: list[tuple] = []
    remaining = list(candidates)
    selected_embs: list[np.ndarray] = [query_vec]  # seed: the query itself

    while len(selected) < top_n and remaining:
        best_mmr_score = -float("inf")
        best_idx = 0

        for i, (final, vis_sim, mid, info) in enumerate(remaining):
            if str(mid) in _index:
                cand_emb = _embeddings_matrix[_index[str(mid)]]
                max_sim_to_selected = max(
                    float(cand_emb @ s) for s in selected_embs
                )
            else:
                max_sim_to_selected = 0.0

            mmr_score = (1 - MMR_LAMBDA) * final - MMR_LAMBDA * max_sim_to_selected
            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_idx = i

        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        _, _, chosen_mid, _ = chosen
        if str(chosen_mid) in _index:
            selected_embs.append(_embeddings_matrix[_index[str(chosen_mid)]])


    # ── 9. Build results ──────────────────────────────────────────────────────
    results = []
    for rank, (final, vis_sim, mal_id, info) in enumerate(selected, start=1):
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
