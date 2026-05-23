"""
Phase 3: Preference Encoder
Converts a free-text preference string (e.g. "dark psychological thriller")
into:
  1. A set of MAL genre names to use as a hard/soft filter
  2. An optional CLIP text embedding for visual-mood matching
"""

import re
import os
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel

# Local model path — download once with: python download_model.py
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "clip-vit-base-patch32")

# ── Full MAL genre / theme / demographic list ────────────────────────────────
MAL_GENRES = [
    "Action", "Adventure", "Avant Garde", "Award Winning",
    "Boys Love", "Comedy", "Drama", "Fantasy", "Girls Love",
    "Gourmet", "Horror", "Mystery", "Romance", "Sci-Fi",
    "Slice of Life", "Sports", "Supernatural", "Suspense",
    # Themes
    "Adult Cast", "Anthropomorphic", "CGDCT", "Childcare",
    "Combat Sports", "Crossdressing", "Delinquents", "Detective",
    "Educational", "Erotica", "Gag Humor", "Gore", "Harem",
    "Hentai", "High Stakes Game", "Historical", "Idols",
    "Isekai", "Iyashikei", "Love Polygon", "Magical Sex Shift",
    "Mahou Shoujo", "Martial Arts", "Mecha", "Medical",
    "Military", "Music", "Mythology", "Organized Crime",
    "Otaku Culture", "Parody", "Performing Arts", "Pets",
    "Psychological", "Racing", "Reincarnation", "Reverse Harem",
    "Romantic Subtext", "School", "Showbiz", "Space",
    "Strategy Game", "Super Power", "Survival", "Team Sports",
    "Time Travel", "Vampire", "Video Game", "Visual Arts", "Workplace",
    # Demographics
    "Josei", "Kids", "Seinen", "Shoujo", "Shounen",
]

# Synonym/alias map → canonical MAL genre name
SYNONYMS: dict[str, str] = {
    "dark": "Psychological",
    "thriller": "Suspense",
    "thrilling": "Suspense",
    "magic": "Supernatural",
    "magical": "Supernatural",
    "futuristic": "Sci-Fi",
    "science fiction": "Sci-Fi",
    "scifi": "Sci-Fi",
    "robot": "Mecha",
    "robots": "Mecha",
    "funny": "Comedy",
    "cute": "CGDCT",
    "sad": "Drama",
    "emotional": "Drama",
    "fight": "Action",
    "fighting": "Action",
    "battle": "Action",
    "war": "Military",
    "detective": "Mystery",
    "crime": "Mystery",
    "ghost": "Supernatural",
    "demon": "Supernatural",
    "demons": "Supernatural",
    "slice of life": "Slice of Life",
    "school life": "School",
    "high school": "School",
    "romance": "Romance",
    "love": "Romance",
    "cooking": "Gourmet",
    "food": "Gourmet",
    "time travel": "Time Travel",
    "isekai": "Isekai",
    "reincarnation": "Reincarnation",
    "fantasy": "Fantasy",
    "sport": "Sports",
    "horror": "Horror",
    "scary": "Horror",
    "music": "Music",
    "adventure": "Adventure",
    "comedy": "Comedy",
    "mystery": "Mystery",
    "gore": "Gore",
    "violent": "Gore",
    "violence": "Gore",
    "military": "Military",
    "samurai": "Historical",
    "ninja": "Action",
    "space": "Space",
    "alien": "Sci-Fi",
    "cyberpunk": "Sci-Fi",
    "vampire": "Vampire",
    "zombies": "Horror",
    "zombie": "Horror",
    "mecha": "Mecha",
    "psychological": "Psychological",
    "survival": "Survival",
    "game": "High Stakes Game",
    "power": "Super Power",
    "superpower": "Super Power",
    "super power": "Super Power",
}

# Build a lowercase lookup: phrase → canonical genre
_GENRE_LOWER = {g.lower(): g for g in MAL_GENRES}


def extract_genres(preference_text: str) -> set[str]:
    """
    Extract MAL genre names from free-text preference.
    Checks exact genre names + alias synonyms.
    Returns a set of canonical MAL genre strings.
    """
    text = preference_text.lower()
    found = set()

    # 1. Exact match against MAL genres (multi-word first, then single word)
    for genre_lower, genre_canonical in _GENRE_LOWER.items():
        pattern = r'\b' + re.escape(genre_lower) + r'\b'
        if re.search(pattern, text):
            found.add(genre_canonical)

    # 2. Synonym expansion
    for alias, canonical in SYNONYMS.items():
        pattern = r'\b' + re.escape(alias) + r'\b'
        if re.search(pattern, text):
            found.add(canonical)

    return found


# ── CLIP text encoder (lazy-loaded) ─────────────────────────────────────────
_clip_model = None
_clip_processor = None


def _get_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        if not os.path.isdir(MODEL_PATH):
            raise FileNotFoundError(
                f"Local CLIP model not found at: {MODEL_PATH}\n"
                f"Please run: python download_model.py"
            )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _clip_model = CLIPModel.from_pretrained(MODEL_PATH, local_files_only=True).to(device)
        _clip_processor = CLIPProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
        _clip_model.eval()
    return _clip_model, _clip_processor


def encode_preference_text(preference_text: str) -> np.ndarray:
    """
    Encode the preference string as a 512-d CLIP text embedding (L2-normalised).
    Used for visual-mood matching alongside the image embeddings.
    """
    model, processor = _get_clip()
    device = next(model.parameters()).device

    inputs = processor(text=[preference_text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_out = model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        pooled = text_out.pooler_output
        text_feat = model.text_projection(pooled)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    return text_feat.cpu().numpy()[0]  # shape: (512,)



class EncodedPreference:
    def __init__(self, raw_text: str):
        self.raw_text = raw_text
        self.genres: set[str] = extract_genres(raw_text)
        self.text_embedding: np.ndarray = encode_preference_text(raw_text)

    def __repr__(self):
        return f"EncodedPreference(genres={self.genres}, text='{self.raw_text}')"


if __name__ == "__main__":
    pref = EncodedPreference("dark psychological thriller with action")
    print(f"Extracted genres: {pref.genres}")
    print(f"Text embedding shape: {pref.text_embedding.shape}")
