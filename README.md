# 🎌 Judging Anime By Its Cover

A multimodal anime recommendation engine that uses **CLIP visual embeddings** to find anime with similar art styles and aesthetic vibes, blended with free-text mood preferences and a **genre co-occurrence correlation system**.

> *"Don't judge a book by its cover" — but we absolutely will judge anime by theirs.*

---

## ✨ How It Works

1. **You provide** a list of anime you've watched (CSV, Excel, or terminal input) and optionally a mood/preference (`"dark psychological thriller"`)
2. **The system** builds a *visual taste vector* from the cover images of your watched anime using CLIP
3. **Worth Level weighting** (optional) — High-rated anime influence the query more than Low-rated ones
4. **Searched against** 30,000+ anime cover embeddings via cosine similarity
5. **Genre injection** — anime matching your requested genres are pulled directly into the candidate pool, even if they don't visually resemble your average taste
6. **Genre relevance scored** using a co-occurrence correlation table built from 30k anime — soft-matching related genres (e.g. `Suspense` and `Psychological` both score high for "horror")
7. **Re-ranked** by visual similarity, MAL score, popularity, and genre relevance
8. **Post-filtered** to exclude anything already in your watched list

---

## 🖥️ Demo

```bash
# Using a CSV watchlist
python recommend.py --input Ani.csv --preference "dark psychological"

# Using terminal input
python recommend.py --input "Death Note, Code Geass, Monster" --preference "mind games thriller"

# Multi-genre cross-filtering
python recommend.py --input Ani.csv --preference "Romance Horror" --top-n 10

# Web GUI (opens in browser at http://localhost:7860)
python gui.py
```

### Sample Output
```
╔═════╤═══════════════════════╤═════════╤═══════════╤══════════════════════╤══════════╗
║  #  │ Title                 │   MAL   │  Members  │ Genres               │  Visual  ║
║     │                       │  Score  │           │                      │   Sim    ║
╟─────┼───────────────────────┼─────────┼───────────┼──────────────────────┼──────────╢
║  1  │ Psycho-Pass           │  8.33   │ 1,751,367 │ Sci-Fi, Suspense     │  0.821   ║
║  2  │ Monster               │  8.70   │   708,412 │ Drama, Mystery,      │  0.819   ║
║  3  │ Talentless Nana       │  7.17   │   380,463 │ Suspense             │  0.812   ║
╚═════╧═══════════════════════╧═════════╧═══════════╧══════════════════════╧══════════╝
```

---

## 🚀 Setup

### 1. Clone the repository
```bash
git clone https://github.com/ChaitanyaParate/Judging-Anime-By-Its-Cover.git
cd Judging-Anime-By-Its-Cover
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install torch torchvision transformers pillow pandas openpyxl \
            rapidfuzz rich numpy aiohttp tqdm gradio
```

### 4. Download the CLIP model (one-time, ~1.2 GB)
```bash
python -c "
from transformers import CLIPModel, CLIPProcessor
model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14')
processor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')
model.save_pretrained('models/clip-vit-large-patch14')
processor.save_pretrained('models/clip-vit-large-patch14')
print('Done!')
"
```

### 5. Download the dataset

All required files are hosted on **Google Drive** — including precomputed embeddings, so you can skip the 55-minute embedding step entirely:

📁 **[Google Drive — Anime Cover Image and Metadata](https://drive.google.com/drive/folders/1uK-QmsqDfnumBUYL23d8LFKXycOW-AWY?usp=sharing)**

| File | Size | Description |
|---|---|---|
| `anime_data.db` | 8.6 MB | SQLite metadata for 30k+ anime |
| `cover_embeddings.npy` | 92.3 MB | Precomputed CLIP embeddings (skip step 6!) |
| `covers.zip` | 2.7 GB | 30k+ anime cover images |

> `embedding_index.json` (maps mal_id → row index, ~450 KB) is **auto-generated** the first time you run `recommend.py`. No download needed.

**Option A — gdown CLI (recommended):**
```bash
pip install gdown

# Download all files from the shared folder
gdown --folder https://drive.google.com/drive/folders/1uK-QmsqDfnumBUYL23d8LFKXycOW-AWY

# Extract covers
unzip covers.zip -d covers/
```

**Option B — Manual download:**
1. Open the Drive link above and download all 3 files
2. Place `anime_data.db` and `cover_embeddings.npy` in the project root
3. Extract `covers.zip` → the images should land in `covers/` in the project root

Your project structure should look like:
```
Judging-Anime-By-Its-Cover/
├── covers/
│   ├── 1.jpg
│   ├── 21.jpg
│   └── ... (30,000+ images)
├── anime_data.db
├── cover_embeddings.npy
├── embedding_index.json      ← auto-generated on first run
├── genre_correlation.json    ← auto-generated on first run
├── models/
│   └── clip-vit-large-patch14/
├── recommend.py
├── gui.py
└── ...
```

### 6. ~~Generate CLIP embeddings~~ (skip — already included in Drive)

> `cover_embeddings.npy` from the Drive is the precomputed output of `embed_covers.py`.  
> Only run `embed_covers.py` if you re-scrape the database with `mal_scraper.py`.

### 7. Run your first recommendation!
```bash
# CLI
python recommend.py --input "Attack on Titan, Death Note" --preference "dark action"

# Web GUI
python gui.py
# → open http://localhost:7860
```

---

## 🖱️ Web GUI

A browser-based interface is included via **Gradio**:

```bash
python gui.py
```

Features:
- Paste anime titles **or** upload your watchlist CSV/Excel directly
- Free-text preference input with detected genre display
- Slider for number of recommendations (3–20)
- Cover image gallery output
- Markdown results table with scores and genres
- Quick-start examples built in

---

## 📁 Input Format

### CSV / Excel
Your file should have a column named `anime`, `title`, `name`, or `show`. Optionally include a `Worth Level` column:

| No. | Anime | Ep. No. | Worth Level |
|-----|-------|---------|-------------|
| 1 | Death Note | 37 | High |
| 2 | Sword Art Online | 96 | Medium |
| 3 | Gamers | 12 | Low |

Worth Level weights: `High = 1.5×`, `Medium = 1.0×`, `Low = 0.3×`

### Terminal input
```bash
python recommend.py --input "Naruto, Bleach, One Piece" --preference "long shonen epic"
```

---

## ⚙️ Scoring Formula

The final score is a weighted sum of four components:

| Component | Weight | Description |
|---|---|---|
| Visual Match | 45% | CLIP cosine similarity of cover aesthetics |
| MAL Score | 28% | Community rating from MyAnimeList |
| Popularity | 17% | Number of MAL members |
| Genre Relevance | 10% | Co-occurrence correlation score (0 if no genre filter or below threshold) |

The **query vector** is: `70% visual taste vector + 30% CLIP text embedding of your preference`

### 🔗 Genre Relevance Scoring

Genre matching goes beyond exact tag lookup. The system builds a **co-occurrence correlation table** from all 30k anime in the database:

- `corr[(G, F)]` = P(genre G | genre F) — how often genre G appears when genre F is present
- For each requested genre, all of the anime's genres contribute a soft correlation score
- Scores are normalized per anime and aggregated using **geometric mean** across requested genres — so an anime must score reasonably on **all** requested genres, not just one
- Anime scoring below `0.13` are hard-zeroed (no genre contribution to final score)

**Demographic genres** (Josei, Seinen, Shoujo, Shounen, Kids) use the **reversed** correlation direction — asking "given this anime has Drama, how likely is it to be Josei?" rather than "what genres appear in Josei anime?", which prevents pure Drama/Romance anime from incorrectly scoring high.

**Genre injection** ensures niche genre anime (e.g. Josei, Racing) always enter the candidate pool even if they look visually different from the user's average taste.

### Supported preference keywords (examples)
`horror`, `isekai`, `mind games`, `thriller`, `romance`, `sports`, `competition`, `mecha`, `military`, `vampire`, `psychological`, `slice of life`, `sci-fi`, `josei`, `seinen`, `shounen`, `ecchi`, `cars/racing`, `surreal`, `police`, `samurai`, `yaoi`, `yuri`, `gore`, `magic`, `harem`, `historical`, `mystery`, and more.

---

## 🔧 CLI Options

```
python recommend.py --input <source> [options]

Arguments:
  --input, -i       CSV file, Excel file, or comma-separated anime titles
  --preference, -p  Free-text mood/genre preference (e.g. "dark psychological")
  --top-n, -n       Number of recommendations (default: 6)
  --candidates      Visual candidates before re-ranking (default: auto-scaled)
```

---

## 📂 Project Structure

| File | Description |
|---|---|
| `recommend.py` | CLI entrypoint — parses args, runs pipeline, prints table + cover grid |
| `gui.py` | Gradio web GUI wrapping the same pipeline |
| `recommender.py` | Core engine — embeddings, MMR ranking, genre correlation scoring |
| `input_parser.py` | Fuzzy-matches user anime titles to MAL IDs via SQLite |
| `preference_encoder.py` | Maps free-text preference to genre tags + CLIP text embedding |
| `embed_covers.py` | One-time script to precompute CLIP embeddings for all covers |
| `mal_scraper.py` | Async scraper for MyAnimeList metadata via Jikan API |
| `anime_data.db` | SQLite database of 30k+ anime (title, score, genres, image URL) |
| `cover_embeddings.npy` | Precomputed 768-d CLIP embeddings for all covers |
| `embedding_index.json` | Maps mal_id → row index in the embedding matrix (auto-generated) |
| `genre_correlation.json` | Genre co-occurrence table (auto-generated / auto-updated) |

---

## 🗄️ Regenerating the Database

If you want to refresh the anime database or scrape new entries:
```bash
python mal_scraper.py --all        # full scrape (~30k anime, slow)
python mal_scraper.py --missing    # only fill in entries missing from DB
```
This automatically rebuilds `genre_correlation.json` after scraping.  
Then re-run `embed_covers.py` to update the embeddings:
```bash
python embed_covers.py
```

---

## 🛠️ Tech Stack

- **[CLIP (ViT-Large/14)](https://github.com/openai/CLIP)** — Visual + text embedding backbone
- **[Jikan API](https://jikan.moe)** — Unofficial MAL REST API for metadata scraping
- **[Gradio](https://gradio.app)** — Web GUI framework
- **NumPy** — Fast cosine similarity over 30k embeddings
- **rapidfuzz** — Fuzzy title matching for CSV/terminal parsing
- **rich** — Terminal UI tables and progress bars
- **aiohttp** — Async HTTP for the MAL scraper
- **SQLite** — Local anime metadata store

---

## 📊 Dataset

| | Details |
|---|---|
| Anime in DB | 30,000+ |
| Cover images | ~30,000 JPGs |
| Database size | 8.6 MB |
| Cover images size | 2.7 GB (`covers.zip`) |
| Embedding size | 92.3 MB (`cover_embeddings.npy`) |

📁 **[Download from Google Drive](https://drive.google.com/drive/folders/1uK-QmsqDfnumBUYL23d8LFKXycOW-AWY?usp=sharing)**

> Data sourced from [MyAnimeList](https://myanimelist.net) via the [Jikan API](https://jikan.moe).  
> This project is non-commercial and for educational/portfolio purposes only.

---

## 📄 License

MIT License — see [LICENSE](LICENSE)
