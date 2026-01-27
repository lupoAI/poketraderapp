"""Dump prime 10 righe per ogni tabella da storage/db/tcgdex_cards.db.

Uso:
    python scripts/dump_tcgdex_db_top10.py

Output:
    storage/db/tcgdex_cards_top10.json

Nota: lo script non modifica il DB.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "storage" / "db" / "tcgdex_cards.db"
OUT_PATH = ROOT / "storage" / "db" / "tcgdex_cards_top10.json"


def _list_tables(con: sqlite3.Connection) -> List[str]:
    cur = con.cursor()
    rows = cur.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type='table'
          AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
    ).fetchall()
    return [r[0] for r in rows]


def _fetch_top10(con: sqlite3.Connection, table: str) -> Dict[str, Any]:
    cur = con.cursor()

    # Colonne
    col_rows = cur.execute(f"PRAGMA table_info({table});").fetchall()
    cols = [r[1] for r in col_rows]

    # Righe (prime 10)
    data_rows = cur.execute(f"SELECT * FROM {table} LIMIT 10;").fetchall()

    def _jsonify(v: Any) -> Any:
        if isinstance(v, (bytes, bytearray)):
            # Evita blob enormi: salva dimensione e un piccolo preview.
            b = bytes(v)
            return {"__blob__": True, "len": len(b), "head_hex": b[:32].hex()}
        return v

    rows: List[Dict[str, Any]] = []
    for r in data_rows:
        rows.append({cols[i]: _jsonify(r[i]) for i in range(len(cols))})

    return {"columns": cols, "rows": rows}


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"DB non trovato: {DB_PATH}")

    con = sqlite3.connect(str(DB_PATH))
    try:
        tables = _list_tables(con)
        out: Dict[str, Any] = {
            "db_path": str(DB_PATH),
            "tables": {},
        }

        for t in tables:
            out["tables"][t] = _fetch_top10(con, t)

        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        con.close()


if __name__ == "__main__":
    main()

