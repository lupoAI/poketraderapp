"""
Script to retrieve information for all Pokémon cards using the PokeData API.

This script collects every available set, then iterates through each set to
retrieve the list of cards contained within it. For each unique card ID
discovered it performs a request to the pricing endpoint to obtain detailed
card information. Because the API uses credits and enforces rate limits,
the implementation includes a simple rate‑limiting mechanism based on
``asyncio`` to avoid flooding the service. When the server signals that
requests are being throttled (HTTP status 429) the client honours the
``Retry‑After`` header and pauses before retrying the same call. Results are
written to a JSON file when finished.

To run this script you need a valid Pokedata API key. Set the
``POKEDATA_API_KEY`` environment variable with your key before running the
script or replace the placeholder in the code. Be aware that large runs
may consume a significant number of credits because every list and detail
request has an associated cost. Always review the Pokedata terms of service
and ensure your use of the API is compliant.

Usage::

    python pokedata_fetch.py --output all_cards.json

You can adjust the concurrency and delay parameters from the command line
depending upon how aggressively you wish to use the API. Higher
concurrency will fetch data faster at the risk of hitting rate limits.
Lower concurrency reduces the chance of throttling but will take longer.
"""

import asyncio
import json
import os
import sys
from argparse import ArgumentParser

import aiohttp

API_BASE = "https://www.pokedata.io"


async def fetch_json(
    session: aiohttp.ClientSession,
    url: str,
    *,
    headers: dict[str, str],
    delay: float,
    retries: int = 5,
) -> dict:
    """Fetch JSON from the given URL with retry and simple rate limiting.

    If the server returns a 429 status code the function respects the
    ``Retry‑After`` header and sleeps for that amount of time before
    repeating the request. A small fixed delay is also inserted before
    every request to avoid spikey traffic patterns.

    Args:
        session: The aiohttp session to use for the request.
        url: Full URL to query.
        headers: HTTP headers including the Bearer token.
        delay: Time in seconds to wait before each request. This helps
            smooth out the rate of requests.
        retries: Number of times to retry on transient errors or 429s.

    Returns:
        Parsed JSON response.

    Raises:
        aiohttp.ClientError: If the request fails after the specified
            number of retries.
    """
    attempt = 0
    while True:
        attempt += 1
        # insert a small delay to prevent burst traffic
        await asyncio.sleep(delay)
        try:
            async with session.get(url, headers=headers) as resp:
                if resp.status == 429:
                    # Respect server requested retry delay if provided.
                    retry_after = resp.headers.get("Retry-After")
                    wait_time = float(retry_after) if retry_after else 60.0
                    print(
                        f"Received 429 Too Many Requests for {url}, waiting {wait_time} seconds before retrying...",
                        file=sys.stderr,
                    )
                    await asyncio.sleep(wait_time)
                    if attempt <= retries:
                        continue
                    else:
                        raise aiohttp.ClientError(
                            f"Exceeded retry count after 429 response for {url}"
                        )
                resp.raise_for_status()
                return await resp.json()
        except (aiohttp.ClientError, aiohttp.ServerTimeoutError) as exc:
            if attempt > retries:
                raise
            backoff = delay * attempt * 2
            print(
                f"Error fetching {url} (attempt {attempt}/{retries}): {exc}, "
                f"sleeping {backoff:.2f} seconds before retrying...",
                file=sys.stderr,
            )
            await asyncio.sleep(backoff)


async def gather_sets(
    session: aiohttp.ClientSession, headers: dict[str, str], delay: float
) -> list[dict]:
    """Retrieve a list of all sets.

    Returns a list of set objects as returned by the API. The optional
    language parameter is omitted here so both English and Japanese sets
    will be returned. You can modify the query string to filter by
    language if desired.
    """
    url = f"{API_BASE}/v0/sets"
    print(f"Fetching sets from {url} ...")
    data = await fetch_json(session, url, headers=headers, delay=delay)
    if not isinstance(data, list):
        raise ValueError(f"Unexpected response format for sets: {data}")
    return data


async def gather_cards_in_set(
    session: aiohttp.ClientSession,
    set_id: int,
    headers: dict[str, str],
    delay: float,
) -> list[dict]:
    """Retrieve the list of card objects for a specific set ID."""
    url = f"{API_BASE}/v0/set?set_id={set_id}"
    data = await fetch_json(session, url, headers=headers, delay=delay)
    if not isinstance(data, list):
        raise ValueError(f"Unexpected response format for cards in set {set_id}: {data}")
    return data


async def gather_card_detail(
    session: aiohttp.ClientSession,
    card_id: int,
    headers: dict[str, str],
    delay: float,
) -> dict:
    """Retrieve detailed information for a card via the pricing endpoint."""
    url = f"{API_BASE}/v0/pricing?id={card_id}&asset_type=CARD"
    return await fetch_json(session, url, headers=headers, delay=delay)


async def main_async(args) -> None:
    """
    Main asynchronous entry point.

    Besides orchestrating the retrieval of set and card data, this function
    persists card pricing information in a local SQLite database. For each
    card ID discovered, it will check whether a record exists for the
    current date before making a pricing API call. This prevents
    unnecessarily consuming API credits and respects rate limits.
    """
    api_key = os.getenv("POKEDATA_API_KEY", "YOUR_API_KEY_HERE")
    if not api_key or api_key == "YOUR_API_KEY_HERE":
        raise SystemExit(
            "No API key provided. set the POKEDATA_API_KEY environment variable or edit the script."
        )
    headers = {"Authorization": f"Bearer {api_key}"}

    # Open database connection and ensure table exists.
    import sqlite3
    from datetime import date

    today_str = date.today().isoformat()
    conn = sqlite3.connect(args.db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS card_prices (
            card_id    INTEGER NOT NULL,
            price_date TEXT NOT NULL,
            data       TEXT NOT NULL,
            PRIMARY KEY (card_id, price_date)
        )
        """
    )
    conn.commit()

    # Helper to check if a price for today exists.
    def has_price(card_id: int) -> bool:
        cur = conn.execute(
            "SELECT 1 FROM card_prices WHERE card_id = ? AND price_date = ? LIMIT 1",
            (card_id, today_str),
        )
        return cur.fetchone() is not None

    # Helper to save a price record.
    def save_price(card_id: int, data_json: str) -> None:
        conn.execute(
            "INSERT OR REPLACE INTO card_prices (card_id, price_date, data) VALUES (?, ?, ?)",
            (card_id, today_str, data_json),
        )
        conn.commit()

    # Create a single session for all requests to leverage connection pooling.
    connector = aiohttp.TCPConnector(limit_per_host=args.concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Step 1: fetch all sets
        sets = await gather_sets(session, headers, args.delay)
        set_ids = [s.get("id") for s in sets if isinstance(s.get("id"), int)]
        print(f"Discovered {len(set_ids)} sets")

        # Step 2: fetch cards for each set. We limit concurrency here by
        # using asyncio.Semaphore to avoid saturating the API.
        card_ids: set[int] = set()

        sem = asyncio.Semaphore(args.concurrency)

        async def process_set(sid: int) -> None:
            async with sem:
                try:
                    cards = await gather_cards_in_set(session, sid, headers, args.delay)
                    for card in cards:
                        cid = card.get("id")
                        if isinstance(cid, int):
                            card_ids.add(cid)
                except Exception as e:
                    # Log and continue
                    print(f"Error processing set {sid}: {e}", file=sys.stderr)

        await asyncio.gather(*(process_set(sid) for sid in set_ids))
        print(f"Discovered {len(card_ids)} unique card IDs")

        # Step 3: fetch details for each card unless already cached for today.
        async def process_card(cid: int) -> None:
            async with sem:
                if has_price(cid):
                    # Skip API call if we already have today's price for this card.
                    print(f"Skipping card {cid}, already have today's data", file=sys.stderr)
                    return
                try:
                    detail = await gather_card_detail(session, cid, headers, args.delay)
                    # Persist as JSON string
                    save_price(cid, json.dumps(detail))
                    print(f"Saved data for card {cid}")
                except Exception as e:
                    print(f"Error retrieving card {cid}: {e}", file=sys.stderr)

        await asyncio.gather(*(process_card(cid) for cid in card_ids))

    conn.close()



def parse_arguments() -> ArgumentParser:
    parser = ArgumentParser(
        description="Download all Pokémon card details via the Pokedata API"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="all_cards.json",
        help="Filename to write the resulting JSON data",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Maximum number of concurrent requests to make",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.2,
        help="Delay in seconds between requests to reduce the chance of hitting rate limits",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=r".\storage\pokedata_cards.db",
        help=(
            "SQLite database file to store card pricing data. "
            "If the file does not exist it will be created."
        ),
    )
    return parser


def main() -> None:
    parser = parse_arguments()
    args = parser.parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("Interrupted by user")


if __name__ == "__main__":
    main()