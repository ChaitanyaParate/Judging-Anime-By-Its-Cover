import asyncio
import aiohttp
import sqlite3
import os
import argparse
import time
from tqdm.asyncio import tqdm
from recommender import _build_genre_correlation

DB_NAME = "anime_data.db"
COVERS_DIR = "covers"
API_URL = "https://api.jikan.moe/v4/anime"

# Rate limits
API_RATE_LIMIT = 3  # requests per second (Jikan API limit)
IMAGE_CONCURRENCY = 30  # concurrent image downloads

def setup_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS anime
                 (mal_id INTEGER PRIMARY KEY,
                  title TEXT,
                  title_english TEXT,
                  title_japanese TEXT,
                  score REAL,
                  scored_by INTEGER,
                  members INTEGER,
                  genres TEXT,
                  image_url TEXT,
                  local_image_path TEXT)''')
    
    # Try to add columns if they don't exist (for existing DBs)
    try:
        c.execute("ALTER TABLE anime ADD COLUMN title_english TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        c.execute("ALTER TABLE anime ADD COLUMN title_japanese TEXT")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    return conn

async def fetch_page(session, page, sem):
    async with sem:
        while True:
            # Respect Jikan rate limit: sleep before making request
            await asyncio.sleep(1.0 / API_RATE_LIMIT)
            url = f"{API_URL}?order_by=score&sort=desc&page={page}"
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:
                    # Rate limit hit, wait and retry
                    await asyncio.sleep(2)
                    continue
                else:
                    print(f"Failed to fetch page {page}: Status {response.status}")
                    return None

async def scrape_metadata():
    conn = setup_db()
    c = conn.cursor()
    
    # First request to get pagination data
    print("Fetching first page to determine total pages...")
    async with aiohttp.ClientSession() as session:
        # Single sem for API to avoid bursts
        api_sem = asyncio.Semaphore(1) 
        first_page = await fetch_page(session, 1, api_sem)
        if not first_page:
            print("Could not fetch first page. Exiting.")
            return

        last_visible_page = first_page['pagination']['last_visible_page']
        print(f"Total pages to scrape: {last_visible_page} (~{last_visible_page * 25} anime)")
        
        # We will process sequentially to strictly adhere to the IP rate limit,
        # but asyncio makes it non-blocking for other tasks (though here it's mostly linear).
        # We process page 1
        process_page_data(first_page['data'], c, conn)

        # Process the rest
        for page in tqdm(range(2, last_visible_page + 1), desc="Scraping Pages"):
            data = await fetch_page(session, page, api_sem)
            if data and 'data' in data:
                process_page_data(data['data'], c, conn)

    conn.close()
    print("Metadata scraping complete.")
    print("Rebuilding genre correlation table...")
    _build_genre_correlation()
    print("Genre correlation updated.")

def process_page_data(anime_list, cursor, conn):
    for anime in anime_list:
        mal_id = anime.get('mal_id')
        title = anime.get('title')
        title_english = anime.get('title_english')
        title_japanese = anime.get('title_japanese')
        score = anime.get('score')
        scored_by = anime.get('scored_by')
        members = anime.get('members')
        
        genres_list = [g.get('name') for g in anime.get('genres', [])]
        # Include MAL Themes (Isekai, Time Travel, Psychological, Military...)
        # and Demographics (Shounen, Seinen...) which Jikan returns separately
        themes_list = [t.get('name') for t in anime.get('themes', [])]
        demographics_list = [d.get('name') for d in anime.get('demographics', [])]
        all_tags = genres_list + themes_list + demographics_list
        genres = ", ".join(dict.fromkeys(all_tags))  # deduplicate, preserve order
        
        # Get large image if available, else normal
        images = anime.get('images', {}).get('jpg', {})
        image_url = images.get('large_image_url') or images.get('image_url')

        try:
            cursor.execute('''INSERT OR REPLACE INTO anime 
                              (mal_id, title, title_english, title_japanese, score, scored_by, members, genres, image_url, local_image_path)
                              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT local_image_path FROM anime WHERE mal_id = ?), NULL))''',
                           (mal_id, title, title_english, title_japanese, score, scored_by, members, genres, image_url, mal_id))
        except sqlite3.Error as e:
            print(f"DB Error on mal_id {mal_id}: {e}")
    conn.commit()

async def download_image(session, sem, mal_id, image_url, pbar):
    async with sem:
        if not image_url:
            pbar.update(1)
            return mal_id, None
            
        ext = image_url.split('.')[-1]
        local_path = os.path.join(COVERS_DIR, f"{mal_id}.{ext}")
        
        if os.path.exists(local_path):
            pbar.update(1)
            return mal_id, local_path

        retries = 5
        for attempt in range(retries):
            try:
                await asyncio.sleep(1.5)
                async with session.get(image_url, timeout=15) as response:
                    if response.status == 200:
                        content = await response.read()
                        with open(local_path, 'wb') as f:
                            f.write(content)
                        pbar.update(1)
                        return mal_id, local_path
                    elif response.status in [429, 403]:
                        # Rate limited or forbidden by CDN
                        await asyncio.sleep(2 * (attempt + 1))
                        continue
                    else:
                        break # Other error, just give up
            except Exception as e:
                await asyncio.sleep(1)
                continue
        
        pbar.update(1)
        return mal_id, None

async def download_images():
    if not os.path.exists(COVERS_DIR):
        os.makedirs(COVERS_DIR)

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT mal_id, image_url FROM anime WHERE local_image_path IS NULL")
    records = c.fetchall()
    
    if not records:
        print("No missing images to download.")
        return

    print(f"Found {len(records)} images to download.")
    
    sem = asyncio.Semaphore(IMAGE_CONCURRENCY)
    
    # We will batch updates to SQLite
    async with aiohttp.ClientSession() as session:
        with tqdm(total=len(records), desc="Downloading Images") as pbar:
            tasks = [download_image(session, sem, mal_id, url, pbar) for mal_id, url in records]
            results = await asyncio.gather(*tasks)
            
            # Update DB
            updates = [(path, mal_id) for mal_id, path in results if path]
            c.executemany("UPDATE anime SET local_image_path = ? WHERE mal_id = ?", updates)
            conn.commit()

    conn.close()
    print("Image download complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MyAnimeList Scraper")
    parser.add_argument("--scrape-metadata", action="store_true", help="Scrape anime metadata from Jikan API")
    parser.add_argument("--download-images", action="store_true", help="Download missing cover images")
    parser.add_argument("--all", action="store_true", help="Run both metadata scraping and image downloading")
    args = parser.parse_args()

    if args.scrape_metadata or args.all:
        asyncio.run(scrape_metadata())
    
    if args.download_images or args.all:
        asyncio.run(download_images())
    
    if not any([args.scrape_metadata, args.download_images, args.all]):
        parser.print_help()
