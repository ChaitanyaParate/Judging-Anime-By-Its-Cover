"""
Phase 5: CLI Entrypoint — Anime Recommendation System
"Don't Judge a Book by Its Cover" — but we definitely judge an anime by its cover.

Usage:
  python recommend.py --input "Naruto, Death Note" --preference "dark action"
  python recommend.py --input my_list.csv --preference "sci-fi space adventure" --top-n 6
  python recommend.py --input my_list.xlsx
"""

import argparse
import os
import sys
import shutil

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns
from rich import box

from input_parser import auto_parse, get_all_raw_titles
from preference_encoder import EncodedPreference
from recommender import recommend, Recommendation
from rapidfuzz import fuzz

console = Console()


# ── ASCII banner ──────────────────────────────────────────────────────────────
BANNER = r"""
  ____  _____   ______      ___   _  ___  _  _____ 
 |  _ \|_   _| |  _ \ \    / / \ | |/ _ \| ||_   _|
 | | | | | |   | | | \ \  / /|  \| | | | | |  | |  
 | |_| | | |   | |_| |\ \/ / | |\  | |_| | |_ | |  
 |____/  |_|   |____/  \__/  |_| \_|\___/ \___||_|  
       J U D G E   B Y   T H E   C O V E R
"""


def print_banner():
    console.print(Panel(
        Text(BANNER, style="bold magenta", justify="center"),
        border_style="bright_magenta",
        padding=(0, 2),
    ))


def print_liked_anime(entries):
    table = Table(
        title="✅ Your Liked Anime (Input)",
        box=box.ROUNDED,
        border_style="cyan",
        show_lines=True,
    )
    table.add_column("Title", style="bold white")
    table.add_column("English Title", style="cyan")
    table.add_column("MAL Score", justify="center", style="yellow")
    table.add_column("Genres", style="dim")

    for e in entries:
        table.add_row(
            e.title,
            e.title_english or "—",
            f"{e.score:.2f}" if e.score else "N/A",
            e.genres or "—",
        )

    console.print(table)
    console.print()


def print_recommendations(results: list[Recommendation], preference: str | None):
    title_text = "🎯  Top Anime Recommendations For You"
    if preference:
        title_text += f"  •  [italic dim]{preference}[/]"

    table = Table(
        title=title_text,
        box=box.DOUBLE_EDGE,
        border_style="bright_magenta",
        show_lines=True,
        padding=(0, 1),
    )

    table.add_column("#", justify="center", style="bold bright_magenta", width=3)
    table.add_column("Title", style="bold white", min_width=22)
    table.add_column("Japanese Title", style="dim cyan", min_width=18)
    table.add_column("MAL\nScore", justify="center", style="bold yellow", width=7)
    table.add_column("Members", justify="right", style="green", width=9)
    table.add_column("Genres", style="dim", min_width=28)
    table.add_column("Visual\nSim", justify="center", style="blue", width=8)

    for r in results:
        score_str = f"[bold green]{r.score:.2f}[/]" if r.score >= 8.0 else \
                    f"[yellow]{r.score:.2f}[/]" if r.score >= 7.0 else \
                    f"[dim]{r.score:.2f}[/]"

        members_str = f"{r.members:,}"
        sim_str = f"{r.similarity:.3f}"

        table.add_row(
            str(r.rank),
            r.title_english or r.title,
            r.title_japanese or "—",
            score_str,
            members_str,
            r.genres or "—",
            sim_str,
        )

    console.print(table)
    console.print()

    # Save cover image paths for reference
    console.print("[bold cyan]Cover images of recommendations:[/]")
    for r in results:
        img_path = r.local_image_path
        if img_path and os.path.exists(img_path):
            abs_path = os.path.abspath(img_path)
            console.print(f"  [green]{r.rank}.[/] {r.title_english or r.title}  →  [dim]{abs_path}[/]")
        else:
            console.print(f"  [green]{r.rank}.[/] {r.title_english or r.title}  →  [red]image not found[/]")

    # ── Export a results grid image ───────────────────────────────────────────
    _save_results_grid(results)


def _save_results_grid(results: list[Recommendation]):
    """Save a side-by-side grid of recommendation cover images."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        import math

        images_with_titles = []
        for r in results:
            if r.local_image_path and os.path.exists(r.local_image_path):
                try:
                    img = Image.open(r.local_image_path).convert("RGB")
                    images_with_titles.append((img, r.title_english or r.title, r.score))
                except Exception:
                    pass

        if not images_with_titles:
            return

        # Resize all to uniform size
        W, H = 200, 280
        LABEL_H = 40
        PADDING = 8
        COLS = min(len(images_with_titles), 3)
        ROWS = math.ceil(len(images_with_titles) / COLS)

        canvas_w = COLS * (W + PADDING) + PADDING
        canvas_h = ROWS * (H + LABEL_H + PADDING) + PADDING
        canvas = Image.new("RGB", (canvas_w, canvas_h), color=(18, 18, 24))
        draw = ImageDraw.Draw(canvas)

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11)
            small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        except Exception:
            font = ImageFont.load_default()
            small_font = font

        for i, (img, title, score) in enumerate(images_with_titles):
            col = i % COLS
            row = i // COLS
            x = PADDING + col * (W + PADDING)
            y = PADDING + row * (H + LABEL_H + PADDING)

            img_resized = img.resize((W, H), Image.LANCZOS)
            canvas.paste(img_resized, (x, y))

            # Label background
            draw.rectangle([x, y + H, x + W, y + H + LABEL_H], fill=(30, 30, 40))
            title_short = title if len(title) <= 22 else title[:19] + "..."
            draw.text((x + 4, y + H + 4), title_short, fill=(220, 220, 255), font=font)
            score_str = f"★ {score:.2f}" if score else ""
            draw.text((x + 4, y + H + 22), score_str, fill=(255, 215, 0), font=small_font)

            # Rank badge
            draw.ellipse([x + 4, y + 4, x + 24, y + 24], fill=(160, 32, 240))
            draw.text((x + 9, y + 6), str(i + 1), fill=(255, 255, 255), font=font)

        os.makedirs("results", exist_ok=True)
        out_path = "results/recommendations.jpg"
        canvas.save(out_path, quality=92)
        console.print(f"\n[bold green]✔[/] Results grid saved → [cyan]{os.path.abspath(out_path)}[/]\n")

    except Exception as e:
        console.print(f"[dim yellow]  (Could not save results grid: {e})[/]")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        prog="recommend.py",
        description="🎌 Anime Recommendation System — powered by CLIP visual embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python recommend.py --input "Naruto, Death Note, Attack on Titan"
  python recommend.py --input my_list.csv --preference "dark psychological thriller"
  python recommend.py --input my_list.xlsx --preference "sci-fi space adventure" --top-n 6
  python recommend.py --input "Cowboy Bebop, Trigun" --preference "nostalgic 90s action" --top-n 5
        """,
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Anime you like: comma-separated names, or path to a .csv/.xlsx file",
    )
    parser.add_argument(
        "--preference", "-p",
        default=None,
        help="What type of anime you want (free text, e.g. 'dark psychological thriller')",
    )
    parser.add_argument(
        "--top-n", "-n",
        type=int,
        default=6,
        help="Number of recommendations to return (default: 6)",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=0,
        help="Visual candidates before re-ranking (default: auto-scaled by input size)",
    )

    args = parser.parse_args()

    print_banner()

    # ── Step 1: Parse input ───────────────────────────────────────────────────
    console.rule("[bold cyan]Step 1 — Parsing Your Anime List[/]")
    console.print(f"  Input: [bold]{args.input}[/]\n")

    liked_anime = auto_parse(args.input)

    if not liked_anime:
        console.print("[bold red]✗ Could not resolve any anime from your input. Please check the titles.[/]")
        sys.exit(1)

    print_liked_anime(liked_anime)
    console.print(f"  [green]✔[/] Resolved [bold]{len(liked_anime)}[/] anime from input.\n")

    # ── Step 2: Encode preference ─────────────────────────────────────────────
    preference_embed = None
    genre_filter = None

    if args.preference:
        console.rule("[bold cyan]Step 2 — Encoding Preference[/]")
        console.print(f"  Preference: [bold italic]{args.preference}[/]\n")

        pref = EncodedPreference(args.preference)
        preference_embed = pref.text_embedding
        genre_filter = pref.genres if pref.genres else None

        if genre_filter:
            console.print(f"  [green]✔[/] Detected genres: [bold yellow]{', '.join(sorted(genre_filter))}[/]")
        else:
            console.print("  [dim]No specific genres detected — using visual preference only.[/]")
        console.print()
    else:
        console.print("[dim]No preference provided — using visual similarity only.[/]\n")

    # ── Step 3: Get recommendations ───────────────────────────────────────────
    console.rule("[bold cyan]Step 3 — Finding Recommendations[/]")
    console.print(f"  Running CLIP similarity search over 30k anime covers...\n")

    # Auto-scale candidates: base + enough to absorb the entire input list.
    # When a genre filter is active, fetch 3× more candidates so genre-tagged
    # anime deeper in visual similarity still get a chance to compete.
    base_candidates = max(300, args.top_n + len(liked_anime) + 100)
    genre_multiplier = 3 if genre_filter else 1
    n_candidates = args.candidates if args.candidates > 0 else base_candidates * genre_multiplier

    console.print(f"  [dim]Fetching top {n_candidates} visual candidates before filtering...[/]\n")

    try:
        results = recommend(
            liked_anime=liked_anime,
            preference_text_embed=preference_embed,
            genre_filter=genre_filter,
            top_n=args.top_n + len(liked_anime) + 50,  # fetch extra; post-filter will trim
            n_candidates=n_candidates,
        )
    except FileNotFoundError as e:
        console.print(f"\n[bold red]✗ Error:[/] {e}")
        sys.exit(1)
    except ValueError as e:
        console.print(f"\n[bold red]✗ Error:[/] {e}")
        sys.exit(1)

    if not results:
        console.print("[bold yellow]⚠ No recommendations found. Try relaxing your preference or adding more liked anime.[/]")
        sys.exit(0)

    # ── Step 3b: Post-filter — remove any result already in user's input ─────
    # This catches anime that failed fuzzy parsing but still appear in results.
    raw_input_titles = get_all_raw_titles(args.input)
    EXCLUSION_THRESHOLD = 78  # lenient — just for exclusion, not matching

    def _is_in_input(rec: Recommendation) -> bool:
        """Return True if this recommendation is essentially in the user's input list."""
        candidates = [
            rec.title.lower() if rec.title else "",
            rec.title_english.lower() if rec.title_english else "",
        ]
        for raw in raw_input_titles:
            if not raw or raw.isdigit() or len(raw) <= 2:
                continue
            for cand in candidates:
                if not cand:
                    continue
                score = fuzz.WRatio(raw, cand)
                if score >= EXCLUSION_THRESHOLD:
                    return True
        return False

    filtered = [r for r in results if not _is_in_input(r)]
    skipped = len(results) - len(filtered)
    if skipped > 0:
        console.print(f"  [dim yellow]ℹ Removed {skipped} result(s) already in your input list.[/]\n")
    results = filtered[:args.top_n]  # trim to desired count

    # Re-number ranks after filter
    results = [r._replace(rank=i + 1) for i, r in enumerate(results)]

    if not results:
        console.print("[bold yellow]⚠ All recommendations were already in your input list. Try a smaller input or add --candidates 300.[/]")
        sys.exit(0)

    # ── Step 4: Display results ───────────────────────────────────────────────
    console.rule("[bold magenta]✨ Your Recommendations[/]")
    print_recommendations(results, args.preference)


if __name__ == "__main__":
    main()
