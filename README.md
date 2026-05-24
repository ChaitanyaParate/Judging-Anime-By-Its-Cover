# 🎌 Judging Anime By Its Cover

A multimodal anime recommendation engine that uses **CLIP visual embeddings** to find anime with similar art styles and aesthetic vibes, blended with free-text mood preferences.

> *"Don't judge a book by its cover" — but we absolutely will judge anime by theirs.*

---

## ✨ How It Works

1. **You provide** a list of anime you've watched (CSV, Excel, or terminal input) and optionally a mood/preference (`"dark psychological thriller"`)
2. **The system** builds a *visual taste vector* from the cover images of your watched anime using CLIP
3. **Worth Level weighting** (optional) — High-rated anime influence the query 5× more than Low-rated ones
4. **Searched against** 30,000+ anime cover embeddings via cosine similarity
5. **Re-ranked** by MAL score, popularity, and genre match
6. **Post-filtered** to exclude anything already in your watched list

---

## 🖥️ Demo

```bash
# Using a CSV watchlist
python recommend.py --input Ani.csv --preference "dark psychological"

# Using terminal input
python recommend.py --input "Death Note, Code Geass, Monster" --preference "mind games thriller"

# More recommendations
python recommend.py --input Ani.csv --preference "wholesome romance" --top-n 10
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
pip install torch torchvision transformers pillow pandas openpyxl rapidfuzz rich numpy
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

### 5. Download the dataset from Kaggle

The anime database and 30k cover images are hosted on Kaggle:

📦 **[Anime Cover Image and Metadata — Kaggle Dataset](https://www.kaggle.com/datasets/chaitanyaparate/anime-cover-image-and-metadata)**

**Option A — Kaggle CLI (recommended):**
```bash
# Install Kaggle CLI and set up API token first: https://github.com/Kaggle/kaggle-api
kaggle datasets download -d chaitanyaparate/anime-cover-image-and-metadata
unzip anime-cover-image-and-metadata.zip

# Fix the nested folder structure (Kaggle adds an extra covers/ layer)
mv covers/covers covers_tmp && rm -rf covers && mv covers_tmp covers
```

**Option B — Manual download:**
1. Download from the Kaggle link above
2. Extract the zip
3. Move `covers/covers/` → `covers/` in the project root
4. Place `anime_data.db` in the project root

Your project structure should look like:
```
Judging-Anime-By-Its-Cover/
├── covers/
│   ├── 1.jpg
│   ├── 21.jpg
│   └── ... (30,000+ images)
├── anime_data.db
├── models/
│   └── clip-vit-large-patch14/
├── recommend.py
└── ...
```

### 6. Generate CLIP embeddings (~55 minutes on GPU)
```bash
nohup python embed_covers.py > embed_covers.log 2>&1 &
tail -f embed_covers.log   # monitor progress
```

### 7. Run your first recommendation!
```bash
python recommend.py --input "Attack on Titan, Death Note" --preference "dark action"
```

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

| Component | Weight | Description |
|---|---|---|
| Visual Match | 55% | CLIP cosine similarity of cover aesthetics |
| MAL Score | 28% | Community rating from MyAnimeList |
| Popularity | 17% | Number of MAL members |
| Genre Bonus | +0.25 max | Flat boost if genres match your preference |

The **query vector** itself is: `70% visual taste + 30% CLIP text embedding of your preference`

---

## 🔧 CLI Options

```
python recommend.py --input <source> [options]

Arguments:
  --input, -i       CSV file, Excel file, or comma-separated anime titles
  --preference, -p  Free-text mood/genre preference (e.g. "dark psychological")
  --top-n, -n       Number of recommendations (default: 6)
  --candidates      Visual candidates before re-ranking (default: auto)
```

---

## 🗄️ Regenerating the Database

If you want to refresh the anime database or scrape new entries:
```bash
python mal_scraper.py --all        # full scrape (slow, ~30k anime)
python mal_scraper.py --missing    # only fill in missing entries
```
Then re-run `embed_covers.py` to update the embeddings.

---

## 🛠️ Tech Stack

- **[CLIP (ViT-Large/14)](https://github.com/openai/CLIP)** — Visual + text embedding backbone
- **[Jikan API](https://jikan.moe)** — Unofficial MAL REST API for metadata scraping
- **NumPy** — Fast cosine similarity over 30k embeddings
- **rapidfuzz** — Fuzzy title matching for CSV parsing
- **rich** — Terminal UI tables and progress bars
- **SQLite** — Local anime metadata store

---

## 📊 Dataset

| | Details |
|---|---|
| Anime in DB | 30,000+ |
| Cover images | ~30,000 JPGs |
| Database size | ~8 MB |
| Cover images size | ~2.7 GB |
| Embedding size | ~85 MB (768-d × 30k) |

> Data sourced from [MyAnimeList](https://myanimelist.net) via the [Jikan API](https://jikan.moe).  
> This project is non-commercial and for educational/portfolio purposes only.

---

## 📄 License

MIT License — see [LICENSE](LICENSE)
