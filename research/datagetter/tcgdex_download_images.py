#!/usr/bin/env python3
"""
Download card images and set logos from a local TCGdex database.

This script reads card and set records from a SQLite database created by
``tcgdex_fetch_full.py`` and downloads low‑resolution images for each card
and set.  Card images are obtained by appending a file extension (e.g.,
``.png``) to the ``image_url`` stored in the ``card_data`` table.  Set
logos are downloaded similarly from the ``logo_url`` field in the
``set_data`` table【461440082945182†L88-L90】.  Images are saved into the
specified directory using the card or set identifier as the filename.

Features:

* Supports asynchronous downloading using ``aiohttp`` for efficient
  throughput.  Concurrency is controlled by a semaphore to avoid
  overwhelming the remote server.
* Skips downloads if the target file already exists (unless
  ``--overwrite`` is specified).
* Allows choosing the image format (PNG or JPG) via the
  ``--image-format`` argument; default is PNG.  Note: the TCGdex asset
  service supports ``png`` or ``jpg`` extensions for cards and sets.

Example usage::

    python download_images.py --db-path tcgdex_cards_full.db \
        --images-dir images --image-format png --concurrency 10

This command will create an ``images`` directory (if it does not exist),
read all cards and sets from the database, and download their images and
logos using up to 10 concurrent connections.
"""

import argparse
import asyncio
import os
import sqlite3
from typing import List, Tuple

import aiohttp
from model.config import STORAGE_DIR


async def fetch_and_save(
    session: aiohttp.ClientSession,
    url: str,
    dest_path: str,
    sem: asyncio.Semaphore,
    overwrite: bool,
) -> None:
    """Download a single image and save it to disk.

    Args:
        session: Shared ``aiohttp.ClientSession`` for HTTP requests.
        url: The full URL to download.
        dest_path: Local filesystem path to write the file to.
        sem: Semaphore limiting concurrent downloads.
        overwrite: If True, always download even if the file exists; if
            False, skip if the file is already present.
    """
    if not overwrite and os.path.exists(dest_path):
        return
    async with sem:
        try:
            async with session.get(url) as resp:
                resp.raise_for_status()
                data = await resp.read()
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                with open(dest_path, "wb") as f:
                    f.write(data)
        except Exception as exc:
            # Log the error but do not abort the whole process
            print(f"Error downloading {url}: {exc}")


def load_records(
    db_path: str,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Load card and set image base URLs from the SQLite database.

    Args:
        db_path: Path to the SQLite database.

    Returns:
        A tuple ``(cards, sets)`` where ``cards`` is a list of
        ``(card_id, image_url)`` tuples and ``sets`` is a list of
        ``(set_id, logo_url)`` tuples.  Entries with missing URLs are
        omitted.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # Fetch cards
    cur.execute("SELECT card_id, image_url FROM card_data")
    cards = [row for row in cur.fetchall() if row[1]]
    # Fetch sets
    cur.execute("SELECT set_id, logo_url FROM set_data")
    sets = [row for row in cur.fetchall() if row[1]]
    conn.close()
    return cards, sets


async def main_async(args) -> None:
    # Load records from database
    cards, sets = load_records(args.db_path)
    print(f"Found {len(cards)} cards and {len(sets)} sets to download.")
    # Prepare tasks
    sem = asyncio.Semaphore(args.concurrency)
    # Use image quality provided by CLI (e.g. low, medium, high, original)
    image_quality = args.image_quality

    async with aiohttp.ClientSession() as session:
        tasks = []
        # Card images
        for card_id, base_url in cards:
            url = f"{base_url}/{image_quality}.{args.image_format}"
            # Save under images_dir/cards
            dest_path = os.path.join(args.images_dir, "cards", f"{card_id}.{args.image_format}")
            tasks.append(fetch_and_save(session, url, dest_path, sem, args.overwrite))
        # Set logos
        for set_id, base_url in sets:
            url = f"{base_url}.{args.image_format}"
            dest_path = os.path.join(args.images_dir, "sets", f"{set_id}.{args.image_format}")
            tasks.append(fetch_and_save(session, url, dest_path, sem, args.overwrite))
        # Run all downloads concurrently
        await asyncio.gather(*tasks)
    print("Download complete.")


def parse_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download card images and set logos from a TCGdex database."
        )
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=str(STORAGE_DIR / "db" / "tcgdex_cards.db"),
        help="Path to the SQLite database produced by tcgdex_fetch.py.",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=str(STORAGE_DIR / "images"),
        help="Directory where images will be saved. Subdirectories 'cards' and 'sets' will be created.",
    )
    parser.add_argument(
        "--image-format",
        type=str,
        choices=["png", "jpg"],
        default="jpg",
        help="Image file format to download (png or jpg).",
    )
    parser.add_argument(
        "--image-quality",
        type=str,
        choices=["low", "medium", "high", "original"],
        default="high",
        help="Image quality to request for card images (low, medium, high, original).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Maximum number of concurrent download requests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing image files if they already exist.",
    )
    return parser


def main() -> None:
    args = parse_arguments().parse_args()
    # Create base directory if not present
    os.makedirs(args.images_dir, exist_ok=True)
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("Interrupted by user")


if __name__ == "__main__":
    main()