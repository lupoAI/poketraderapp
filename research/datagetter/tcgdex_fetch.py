#!/usr/bin/env python3
"""
Fetch complete Pokémon Trading Card Game (TCG) data from the **TCGdex REST API**
and persist it in a SQLite database.  This script extends the basic raw
collector by also retrieving set metadata and maintaining tables of unique
cards and sets across runs.

The TCGdex API exposes endpoints for listing cards (`GET /v2/{lang}/cards`),
fetching detailed card objects (`GET /v2/{lang}/cards/{id}`)【70314678094995†L79-L112】, listing
sets (`GET /v2/{lang}/sets`)【283501927614676†L142-L200】 and retrieving full set
information (`GET /v2/{lang}/sets/{id}`)【70314678094995†L79-L207】.  Sets include basic
metadata such as name, logo, symbol and card counts, along with an array of
card identifiers【70314678094995†L116-L180】.  The API does not paginate by
default【283501927614676†L138-L140】, so requests return the full list in a single
response.

This script performs the following steps:

1. Create or open a SQLite database and ensure tables exist for
   `card_data`, `set_data`, `unique_cards` and `unique_sets`.
2. Fetch the list of all sets and update the `unique_sets` table.  For
   each set, fetch its detailed object and store it in `set_data`.  The
   `set_data` table holds a single record per set identifier; subsequent
   runs update the existing record rather than inserting duplicates.
3. Fetch the list of all cards and update the `unique_cards` table.  For
   each card, fetch its detailed object and store it in `card_data`.  The
   `card_data` table holds a single record per card identifier and is
   overwritten on subsequent runs.  If a card fetch returns 404 (for
   example, placeholder IDs such as ``exu-?``), the script skips the
   record.
4. Concurrency and delay parameters control the number of simultaneous
   requests and the base delay between calls to mitigate rate limits.  The
   `fetch_json` helper respects HTTP 429 ``Retry-After`` headers and
   gracefully skips 404 responses.

To run the script:

```
python tcgdex_fetch_full.py --db-path tcgdex_cards_full.db --language en \
    --concurrency 5 --delay 0.2
```

This will populate the database with up-to-date set and card information
while tracking all unique sets and cards.
"""

import asyncio
import json
import sqlite3
import logging
from argparse import ArgumentParser
from datetime import date
from typing import Any, Dict, List, Optional

import aiohttp

API_BASE = "https://api.tcgdex.net/v2"


async def fetch_json(
    session: aiohttp.ClientSession,
    url: str,
    *,
    delay: float,
    retries: int = 5,
) -> Optional[Any]:
    """Retrieve JSON from a URL with basic retry and 404 handling.

    A small delay is applied before each request to smooth out bursts.
    If the server responds with 429 (Too Many Requests), the function
    waits for the number of seconds specified by ``Retry-After`` before
    retrying.  For 404 responses, it returns ``None`` so callers can
    skip missing resources.

    Args:
        session: A live ``aiohttp`` session.
        url: The full URL to request.
        delay: Base delay between requests in seconds.
        retries: Maximum number of retry attempts on failure.

    Returns:
        Parsed JSON on success or ``None`` if the resource does not exist.

    Raises:
        aiohttp.ClientError: If repeated attempts fail due to network
            errors or non-404 HTTP responses after exhausting retries.
    """
    attempt = 0
    while True:
        attempt += 1
        await asyncio.sleep(delay)
        try:
            # Log the outgoing request
            logging.info(f"Request: {url}")
            async with session.get(url) as resp:
                if resp.status == 429:
                    retry_after = resp.headers.get("Retry-After")
                    wait_time = float(retry_after) if retry_after else 60.0
                    print(
                        f"429 Too Many Requests for {url}, retrying after {wait_time} seconds…",
                        flush=True,
                    )
                    await asyncio.sleep(wait_time)
                    if attempt <= retries:
                        continue
                    raise aiohttp.ClientError(
                        f"Exceeded retry count after 429 for {url}"
                    )
                if resp.status == 404:
                    # Some IDs are placeholders (e.g., exu-?) and return 404
                    print(f"404 Not Found for {url}, skipping.", flush=True)
                    return None
                resp.raise_for_status()
                return await resp.json()
        except (aiohttp.ClientError, aiohttp.ServerTimeoutError) as exc:
            if attempt > retries:
                print(f"Error fetching {url}: {exc}", flush=True)
                raise
            backoff = delay * attempt * 2
            print(
                f"Error fetching {url} (attempt {attempt}/{retries}): {exc}; sleeping {backoff:.2f} s before retrying…",
                flush=True,
            )
            await asyncio.sleep(backoff)


def setup_database(conn: sqlite3.Connection) -> None:
    """Initialise the SQLite database with required tables.

    This function creates four tables:

    * ``card_data`` stores a flattened view of each card's detailed
      information.  Columns map to top‑level fields of the Card object
      (e.g., ``category``, ``illustrator``, ``rarity``) and selected nested
      properties from the card's set and variants.  This allows direct
      querying of specific attributes without unpacking a JSON blob.
    * ``set_data`` stores a flattened view of each set's detailed
      information.  Columns correspond to fields of the Set object such
      as ``releaseDate``, ``serie.id``, ``cardCount.total`` and legal
      status.  Nested lists (e.g., ``cards`` and ``boosters``) are
      preserved as JSON strings for completeness.
    * ``unique_cards`` and ``unique_sets`` track the card and set
      identifiers that have been encountered.  They store the brief
      object fields (id, name, etc.) along with ``first_seen_date`` and
      ``last_seen_date``.  A ``raw_json`` column holds the full object for
      auditing but is not intended to be queried directly.
    """
    # Detailed card table.  Store frequently used fields directly and keep the
    # entire card JSON for reference.  See the Card object reference for
    # definitions of each field【566008020801343†L130-L146】【566008020801343†L175-L207】.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS card_data (
            card_id           TEXT PRIMARY KEY,
            local_id          TEXT,
            name              TEXT,
            image         TEXT,
            category          TEXT,
            illustrator       TEXT,
            rarity            TEXT,
            set_id            TEXT,
            set_name          TEXT,
            set_logo_url      TEXT,
            set_symbol_url    TEXT,
            set_total_count   INTEGER,
            set_official_count INTEGER,
            hp                INTEGER,
            types             TEXT,
            evolve_from       TEXT,
            description       TEXT,
            stage             TEXT,
            retreat           INTEGER,
            regulation_mark   TEXT,
            legal_standard    INTEGER,
            legal_expanded    INTEGER,
            variant_normal    INTEGER,
            variant_reverse   INTEGER,
            variant_holo      INTEGER,
            variant_firstEdition INTEGER,
            variant_wPromo    INTEGER,
            updated_at        TEXT,
            pricing_json      TEXT,
            attacks_json      TEXT,
            weaknesses_json   TEXT,
            boosters_json     TEXT,
            raw_json          TEXT
        )
        """
    )
    # Detailed set table.  Store key properties and counts, plus JSON for cards and boosters.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS set_data (
            set_id           TEXT PRIMARY KEY,
            name             TEXT,
            logo_url         TEXT,
            symbol_url       TEXT,
            release_date     TEXT,
            series_id        TEXT,
            series_name      TEXT,
            total_count      INTEGER,
            official_count   INTEGER,
            normal_count     INTEGER,
            reverse_count    INTEGER,
            holo_count       INTEGER,
            firstEd_count    INTEGER,
            legal_standard   INTEGER,
            legal_expanded   INTEGER,
            tcg_online       TEXT,
            cards_json       TEXT,
            boosters_json    TEXT,
            raw_json         TEXT
        )
        """
    )
    # Unique cards table storing card brief information and first/last seen dates.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS unique_cards (
            card_id         TEXT PRIMARY KEY,
            local_id        TEXT,
            name            TEXT,
            image       TEXT,
            first_seen_date TEXT NOT NULL,
            last_seen_date  TEXT NOT NULL,
            raw_json        TEXT
        )
        """
    )
    # Unique sets table storing set brief information and first/last seen dates.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS unique_sets (
            set_id         TEXT PRIMARY KEY,
            name           TEXT,
            logo_url       TEXT,
            symbol_url     TEXT,
            total_count    INTEGER,
            official_count INTEGER,
            first_seen_date TEXT NOT NULL,
            last_seen_date  TEXT NOT NULL,
            raw_json       TEXT
        )
        """
    )
    conn.commit()


async def fetch_cards_list(
    session: aiohttp.ClientSession,
    language: str,
    delay: float,
) -> List[Dict[str, Any]]:
    """Fetch the list of all card brief objects for the specified language."""
    url = f"{API_BASE}/{language}/cards"
    cards = await fetch_json(session, url, delay=delay)
    return cards if isinstance(cards, list) else []


async def fetch_card_details(
    session: aiohttp.ClientSession,
    language: str,
    card_id: str,
    delay: float,
) -> Optional[Dict[str, Any]]:
    """Fetch the detailed card object by card identifier."""
    url = f"{API_BASE}/{language}/cards/{card_id}"
    return await fetch_json(session, url, delay=delay)


async def fetch_sets_list(
    session: aiohttp.ClientSession,
    language: str,
    delay: float,
) -> List[Dict[str, Any]]:
    """Fetch the list of all set brief objects for the specified language."""
    url = f"{API_BASE}/{language}/sets"
    sets = await fetch_json(session, url, delay=delay)
    return sets if isinstance(sets, list) else []


async def fetch_set_details(
    session: aiohttp.ClientSession,
    language: str,
    set_id: str,
    delay: float,
) -> Optional[Dict[str, Any]]:
    """Fetch full set information by set identifier."""
    url = f"{API_BASE}/{language}/sets/{set_id}"
    return await fetch_json(session, url, delay=delay)


def extract_card_columns(card: Dict[str, Any]) -> Dict[str, Any]:
    """Extract selected columns from a full card object for the unique_cards table.

    This helper flattens commonly used properties and leaves the
    remaining structure in ``raw_json``.
    """
    # For the card brief, only capture the identifier, local ID, name and image URL,
    # along with the raw JSON string. Additional fields such as category, rarity
    # and types are present in the full card object but are not stored in the
    # unique_cards table to align with the Card brief specification【214926388313869†L83-L103】.
    return {
        "card_id": card.get("id"),
        "local_id": card.get("localId"),
        "name": card.get("name"),
        "image": card.get("image"),
        "raw_json": json.dumps(card),
    }


def extract_set_columns(set_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract selected columns from a full set object for the unique_sets table."""
    # For the set brief, capture only the identifier, name, logo, symbol and the
    # total and official card counts. Additional fields (release date, series,
    # and counts by rarity) exist on the full Set object but are not included
    # in the Set brief specification【409146615334698†L80-L120】.
    card_count = set_data.get("cardCount", {}) or {}
    return {
        "set_id": set_data.get("id"),
        "name": set_data.get("name"),
        "logo_url": set_data.get("logo"),
        "symbol_url": set_data.get("symbol"),
        "total_count": card_count.get("total"),
        "official_count": card_count.get("official"),
        "raw_json": json.dumps(set_data),
    }

# ---------------------------------------------------------------------------
# Data flattening helpers
#
# These functions transform the nested structures returned by the TCGdex REST
# API into flat dictionaries suitable for insertion into the ``card_data`` and
# ``set_data`` tables.  They extract commonly used fields and coerce values
# into types compatible with SQLite (e.g., booleans -> integers and arrays ->
# comma-separated strings).  Complex nested objects such as pricing,
# attacks, weaknesses, boosters, cards and boosters are serialized as JSON
# strings to preserve the full information while enabling flexible queries.

def flatten_card_data(card: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a full card object into a dictionary matching the card_data schema.

    Args:
        card: A full card record returned by ``GET /v2/{lang}/cards/{id}``.

    Returns:
        A dictionary keyed by column names in the ``card_data`` table.  Nested
        structures are serialized as JSON strings, and boolean values are
        converted to integers for storage in SQLite.
    """
    # Set information may be absent; fall back to empty dict
    set_info = card.get("set", {}) or {}
    set_card_count = set_info.get("cardCount", {}) or {}

    # Variants may be absent; default to empty dict
    variants = card.get("variants", {}) or {}

    # Legalities may be absent
    legal = card.get("legal", {}) or {}

    # Convert types array to a comma-separated string
    types = card.get("types")
    types_str = ",".join(types) if isinstance(types, list) else None

    # Serialize nested objects to JSON strings; None if missing
    pricing_json = json.dumps(card.get("pricing")) if card.get("pricing") is not None else None
    attacks_json = json.dumps(card.get("attacks")) if card.get("attacks") is not None else None
    weaknesses_json = json.dumps(card.get("weaknesses")) if card.get("weaknesses") is not None else None
    boosters_json = json.dumps(card.get("boosters")) if card.get("boosters") is not None else None

    # Helper to convert boolean values to integers
    def bool_to_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        return 1 if bool(value) else 0

    return {
        "card_id": card.get("id"),
        "local_id": card.get("localId"),
        "name": card.get("name"),
        "image": card.get("image"),
        "category": card.get("category"),
        "illustrator": card.get("illustrator"),
        "rarity": card.get("rarity"),
        "set_id": set_info.get("id"),
        "set_name": set_info.get("name"),
        "set_logo_url": set_info.get("logo"),
        "set_symbol_url": set_info.get("symbol"),
        "set_total_count": set_card_count.get("total"),
        "set_official_count": set_card_count.get("official"),
        "hp": card.get("hp"),
        "types": types_str,
        "evolve_from": card.get("evolveFrom"),
        "description": card.get("description"),
        "stage": card.get("stage"),
        "retreat": card.get("retreat"),
        "regulation_mark": card.get("regulationMark"),
        "legal_standard": bool_to_int(legal.get("standard")),
        "legal_expanded": bool_to_int(legal.get("expanded")),
        "variant_normal": bool_to_int(variants.get("normal")),
        "variant_reverse": bool_to_int(variants.get("reverse")),
        "variant_holo": bool_to_int(variants.get("holo")),
        "variant_firstEdition": bool_to_int(variants.get("firstEdition")),
        "variant_wPromo": bool_to_int(variants.get("wPromo")),
        "updated_at": card.get("updated"),
        "pricing_json": pricing_json,
        "attacks_json": attacks_json,
        "weaknesses_json": weaknesses_json,
        "boosters_json": boosters_json,
        "raw_json": json.dumps(card),
    }


def flatten_set_data(set_obj: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a full set object into a dictionary matching the set_data schema.

    Args:
        set_obj: A full set record returned by ``GET /v2/{lang}/sets/{id}``.

    Returns:
        A dictionary keyed by column names in the ``set_data`` table.  Nested
        lists and objects are serialized to JSON strings where appropriate.
    """
    card_count = set_obj.get("cardCount", {}) or {}
    legal = set_obj.get("legal", {}) or {}
    serie = set_obj.get("serie", {}) or {}

    # Serialize lists to JSON strings
    cards_json = json.dumps(set_obj.get("cards")) if set_obj.get("cards") is not None else None
    boosters_json = json.dumps(set_obj.get("boosters")) if set_obj.get("boosters") is not None else None

    return {
        "set_id": set_obj.get("id"),
        "name": set_obj.get("name"),
        "logo_url": set_obj.get("logo"),
        "symbol_url": set_obj.get("symbol"),
        "release_date": set_obj.get("releaseDate"),
        "series_id": serie.get("id"),
        "series_name": serie.get("name"),
        "total_count": card_count.get("total"),
        "official_count": card_count.get("official"),
        "normal_count": card_count.get("normal"),
        "reverse_count": card_count.get("reverse"),
        "holo_count": card_count.get("holo"),
        "firstEd_count": card_count.get("firstEd"),
        "legal_standard": 1 if legal.get("standard") else 0 if legal.get("standard") is not None else None,
        "legal_expanded": 1 if legal.get("expanded") else 0 if legal.get("expanded") is not None else None,
        "tcg_online": set_obj.get("tcgOnline"),
        "cards_json": cards_json,
        "boosters_json": boosters_json,
        "raw_json": json.dumps(set_obj),
    }


async def upsert_unique_card(
    conn: sqlite3.Connection,
    card_cols: Dict[str, Any],
    today_str: str,
    lock: asyncio.Lock,
) -> None:
    """Insert or update a record in the unique_cards table.

    The card is identified by its ``card_id``.  If the record already exists,
    ``last_seen_date`` and the extracted columns are updated; otherwise the
    record is inserted with ``first_seen_date`` and ``last_seen_date`` set
    to ``today_str``.
    """
    card_id = card_cols.get("card_id")
    if not card_id:
        return
    async with lock:
        cur = conn.cursor()
        row = cur.execute(
            "SELECT first_seen_date FROM unique_cards WHERE card_id = ?",
            (card_id,),
        ).fetchone()
        if row:
            # Update existing row
            first_seen = row[0]
            # Prepare update statement: update all columns except first_seen_date
            update_cols = {k: v for k, v in card_cols.items() if k != "card_id"}
            update_cols["last_seen_date"] = today_str
            assignments = ", ".join(f"{col} = ?" for col in update_cols.keys())
            values = list(update_cols.values()) + [card_id]
            cur.execute(
                f"UPDATE unique_cards SET {assignments} WHERE card_id = ?",
                values,
            )
        else:
            # Insert new row
            insert_cols = ["card_id", "first_seen_date", "last_seen_date"] + list(card_cols.keys())
            # Remove duplicates in case card_id is included in card_cols
            insert_cols = list(dict.fromkeys(insert_cols))
            placeholders = ",".join(["?"] * len(insert_cols))
            values = []
            for col in insert_cols:
                if col == "card_id":
                    values.append(card_id)
                elif col == "first_seen_date" or col == "last_seen_date":
                    values.append(today_str)
                else:
                    values.append(card_cols.get(col))
            cols_clause = ", ".join(insert_cols)
            cur.execute(
                f"INSERT INTO unique_cards ({cols_clause}) VALUES ({placeholders})",
                values,
            )
        conn.commit()


async def upsert_unique_set(
    conn: sqlite3.Connection,
    set_cols: Dict[str, Any],
    today_str: str,
    lock: asyncio.Lock,
) -> None:
    """Insert or update a record in the unique_sets table.

    Identified by ``set_id``, updating all extracted columns along with
    first_seen_date and last_seen_date.
    """
    set_id = set_cols.get("set_id")
    if not set_id:
        return
    async with lock:
        cur = conn.cursor()
        row = cur.execute(
            "SELECT first_seen_date FROM unique_sets WHERE set_id = ?",
            (set_id,),
        ).fetchone()
        if row:
            first_seen = row[0]
            update_cols = {k: v for k, v in set_cols.items() if k != "set_id"}
            update_cols["last_seen_date"] = today_str
            assignments = ", ".join(f"{col} = ?" for col in update_cols.keys())
            values = list(update_cols.values()) + [set_id]
            cur.execute(
                f"UPDATE unique_sets SET {assignments} WHERE set_id = ?",
                values,
            )
        else:
            insert_cols = ["set_id", "first_seen_date", "last_seen_date"] + list(set_cols.keys())
            insert_cols = list(dict.fromkeys(insert_cols))
            placeholders = ",".join(["?"] * len(insert_cols))
            values = []
            for col in insert_cols:
                if col == "set_id":
                    values.append(set_id)
                elif col == "first_seen_date" or col == "last_seen_date":
                    values.append(today_str)
                else:
                    values.append(set_cols.get(col))
            cols_clause = ", ".join(insert_cols)
            cur.execute(
                f"INSERT INTO unique_sets ({cols_clause}) VALUES ({placeholders})",
                values,
            )
        conn.commit()


async def update_unique_entry(
    conn: sqlite3.Connection,
    table: str,
    id_value: str,
    today_str: str,
    lock: asyncio.Lock,
) -> None:
    """Insert or update an entry in the unique_cards/unique_sets table.

    If the id_value does not exist, insert with first_seen_date = today.
    Always update last_seen_date to today.
    """
    async with lock:
        cur = conn.cursor()
        # Determine which identifier column to use based on table name.  The
        # unique_cards table stores a "card_id" column; the unique_sets table
        # stores a "set_id" column.
        if table == "unique_cards":
            id_col = "card_id"
        elif table == "unique_sets":
            id_col = "set_id"
        else:
            raise ValueError(f"Unsupported table for unique entry: {table}")
        row = cur.execute(
            f"SELECT 1 FROM {table} WHERE {id_col} = ?",
            (id_value,),
        ).fetchone()
        if row:
            # Update last seen date
            cur.execute(
                f"UPDATE {table} SET last_seen_date = ? WHERE {id_col} = ?",
                (today_str, id_value),
            )
        else:
            # Insert new entry with first and last seen dates
            cur.execute(
                f"INSERT INTO {table} ({id_col}, first_seen_date, last_seen_date) VALUES (?, ?, ?)",
                (id_value, today_str, today_str),
            )
        conn.commit()


async def store_record(
    conn: sqlite3.Connection,
    table: str,
    id_value: str,
    data_json: str,
    lock: asyncio.Lock,
) -> None:
    """Insert or replace a record in the given data table (card_data or set_data).

    Since the data tables no longer track per-day records, this always
    updates the existing record for ``id_value`` or inserts a new one.
    """
    async with lock:
        cur = conn.cursor()
        cur.execute(
            f"INSERT OR REPLACE INTO {table} ({table[:-5]}_id, data) VALUES (?, ?)",
            (id_value, data_json),
        )
        conn.commit()


async def store_card_record(
    conn: sqlite3.Connection,
    card_dict: Dict[str, Any],
    lock: asyncio.Lock,
) -> None:
    """Insert or replace a flattened card record into the ``card_data`` table.

    Args:
        conn: The SQLite connection.
        card_dict: A dictionary produced by :func:`flatten_card_data`.
        lock: An asyncio lock to prevent concurrent writes.
    """
    async with lock:
        cur = conn.cursor()
        cols = list(card_dict.keys())
        placeholders = ",".join(["?"] * len(cols))
        values = [card_dict[col] for col in cols]
        cur.execute(
            f"INSERT OR REPLACE INTO card_data ({','.join(cols)}) VALUES ({placeholders})",
            values,
        )
        conn.commit()


async def store_set_record(
    conn: sqlite3.Connection,
    set_dict: Dict[str, Any],
    lock: asyncio.Lock,
) -> None:
    """Insert or replace a flattened set record into the ``set_data`` table.

    Args:
        conn: The SQLite connection.
        set_dict: A dictionary produced by :func:`flatten_set_data`.
        lock: An asyncio lock to prevent concurrent writes.
    """
    async with lock:
        cur = conn.cursor()
        cols = list(set_dict.keys())
        placeholders = ",".join(["?"] * len(cols))
        values = [set_dict[col] for col in cols]
        cur.execute(
            f"INSERT OR REPLACE INTO set_data ({','.join(cols)}) VALUES ({placeholders})",
            values,
        )
        conn.commit()


async def process_set(
    session: aiohttp.ClientSession,
    conn: sqlite3.Connection,
    language: str,
    set_brief: Dict[str, Any],
    today_str: str,
    delay: float,
    sem: asyncio.Semaphore,
    db_lock: asyncio.Lock,
) -> None:
    """Fetch and store a single set's details, updating unique sets and set_data."""
    set_id = set_brief.get("id")
    if not set_id:
        return
    # Fetch full set details.  Do not update unique_sets until we have the details.
    async with sem:
        details = await fetch_set_details(session, language, set_id, delay)
    if details is None:
        return
    # Upsert into unique_sets with extracted columns
    set_cols = extract_set_columns(details)
    await upsert_unique_set(conn, set_cols, today_str, db_lock)
    # Flatten the full set record and store it in set_data
    flat_set = flatten_set_data(details)
    await store_set_record(conn, flat_set, db_lock)


async def process_card(
    session: aiohttp.ClientSession,
    conn: sqlite3.Connection,
    language: str,
    card_brief: Dict[str, Any],
    today_str: str,
    delay: float,
    sem: asyncio.Semaphore,
    db_lock: asyncio.Lock,
) -> None:
    """Fetch and store a single card's details, updating unique cards and card_data."""
    card_id = card_brief.get("id")
    if not card_id:
        return
    # Fetch full card details.  Only after retrieving details we can update the unique_cards table.
    async with sem:
        details = await fetch_card_details(session, language, card_id, delay)
    if details is None:
        return
    # Upsert into unique_cards with extracted columns
    card_cols = extract_card_columns(details)
    await upsert_unique_card(conn, card_cols, today_str, db_lock)
    # Flatten the full card record and store it in card_data
    flat_card = flatten_card_data(details)
    await store_card_record(conn, flat_card, db_lock)


async def main_async(args) -> None:
    """Main asynchronous entry point handling card and set retrieval."""
    language: str = args.language
    db_path: str = args.db_path
    concurrency: int = args.concurrency
    delay: float = args.delay
    today_str = date.today().isoformat()

    # Initialise database
    conn = sqlite3.connect(db_path, check_same_thread=False)
    setup_database(conn)
    db_lock = asyncio.Lock()

    sem = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit_per_host=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Fetch sets list
        print("Fetching sets list…", flush=True)
        sets_brief = await fetch_sets_list(session, language, delay)
        print(f"Found {len(sets_brief)} sets.", flush=True)
        # Process sets concurrently
        set_tasks = [
            asyncio.create_task(
                process_set(
                    session,
                    conn,
                    language,
                    s,
                    today_str,
                    delay,
                    sem,
                    db_lock,
                )
            )
            for s in sets_brief
        ]
        # Fetch cards list
        print("Fetching cards list…", flush=True)
        cards_brief = await fetch_cards_list(session, language, delay)
        print(f"Found {len(cards_brief)} cards.", flush=True)
        # Process cards concurrently
        card_tasks = [
            asyncio.create_task(
                process_card(
                    session,
                    conn,
                    language,
                    c,
                    today_str,
                    delay,
                    sem,
                    db_lock,
                )
            )
            for c in cards_brief
        ]
        await asyncio.gather(*(set_tasks + card_tasks))
    conn.close()


def parse_arguments() -> ArgumentParser:
    """Parse command-line arguments for the script."""
    parser = ArgumentParser(
        description=(
            "Download all Pokémon TCG cards and sets from the TCGdex REST API and "
            "persist them to separate tables in a SQLite database while "
            "tracking unique entries."
        )
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=r".\storage\tcgdex_cards.db",
        help="SQLite database file. Will be created if it does not exist.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language code for the API (default: 'en').",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Maximum number of concurrent HTTP requests.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Delay in seconds between requests to mitigate rate limits.",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default=r".\storage\tcgdex_queries.log",
        help=(
            "Path to a log file for recording API requests. Set to '-' to log to stdout."
        ),
    )
    return parser


def main() -> None:
    args = parse_arguments().parse_args()
    # Configure logging.  If the user specifies "-" for log_path, log to stdout.
    log_path = getattr(args, "log_path", None)
    if log_path:
        if log_path == "-":
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s %(levelname)s: %(message)s",
            )
        else:
            logging.basicConfig(
                filename=log_path,
                level=logging.INFO,
                format="%(asctime)s %(levelname)s: %(message)s",
            )
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("Interrupted by user")


if __name__ == "__main__":
    main()