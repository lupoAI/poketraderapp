"""Estrae tutte le possibili combinazioni in `variants_detailed` dai record carte.

Per "possibili" intendiamo:
- set di `type`
- set di `size`
- set di `foil` (se presente)
- set di triple uniche (type,size,foil)

Fonte dati:
- preferisce leggere direttamente dal DB `storage/db/tcgdex_cards.db` e il campo `raw_json` della tabella `card_data`
- fallback: legge `storage/db/tcgdex_cards_top10.json`

Output:
- stampa a console un riassunto
- salva anche `storage/db/variants_detailed_values.json`
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "storage" / "db" / "tcgdex_cards.db"
DUMP_PATH = ROOT / "storage" / "db" / "tcgdex_cards_top10.json"
OUT_PATH = ROOT / "storage" / "db" / "variants_detailed_values.json"


def _iter_raw_json_from_db() -> Iterable[str]:
    con = sqlite3.connect(str(DB_PATH))
    try:
        cur = con.cursor()
        # card_data sembra essere la tabella con raw_json (dal dump). Se non esiste, alza.
        cur.execute("SELECT raw_json FROM card_data WHERE raw_json IS NOT NULL")
        for (raw,) in cur.fetchall():
            if raw:
                yield raw
    finally:
        con.close()


def _iter_raw_json_from_dump() -> Iterable[str]:
    d = json.loads(DUMP_PATH.read_text(encoding="utf-8"))
    tables = d.get("tables", {})
    card_data = tables.get("card_data", {})
    rows = card_data.get("rows", [])
    for r in rows:
        raw = r.get("raw_json")
        if raw:
            yield raw


def _normalize_foil(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, str) and v.strip() == "":
        return None
    return str(v)


def main() -> None:
    if DB_PATH.exists():
        raws = list(_iter_raw_json_from_db())
        source = "db"
    elif DUMP_PATH.exists():
        raws = list(_iter_raw_json_from_dump())
        source = "dump"
    else:
        raise FileNotFoundError("Né DB né dump trovati.")

    types: Set[str] = set()
    sizes: Set[str] = set()
    foils: Set[str] = set()
    triples: Set[Tuple[str, str, Optional[str]]] = set()

    with_variants = 0

    for raw in raws:
        try:
            obj = json.loads(raw)
        except Exception:
            continue

        vd = obj.get("variants_detailed")
        if not isinstance(vd, list):
            continue

        with_variants += 1
        for item in vd:
            if not isinstance(item, dict):
                continue
            t = item.get("type")
            s = item.get("size")
            f = _normalize_foil(item.get("foil"))
            if isinstance(t, str):
                types.add(t)
            if isinstance(s, str):
                sizes.add(s)
            if f is not None:
                foils.add(f)
            if isinstance(t, str) and isinstance(s, str):
                triples.add((t, s, f))

    out: Dict[str, Any] = {
        "source": source,
        "counts": {
            "raw_json_rows": len(raws),
            "cards_with_variants_detailed": with_variants,
            "unique_types": len(types),
            "unique_sizes": len(sizes),
            "unique_foils": len(foils),
            "unique_entries": len(triples),
        },
        "types": sorted(types),
        "sizes": sorted(sizes),
        "foils": sorted(foils),
        "entries": [
            {"type": t, "size": s, **({"foil": f} if f is not None else {})}
            for (t, s, f) in sorted(triples, key=lambda x: (x[0], x[1], x[2] or ""))
        ],
    }

    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("variants_detailed values extracted from:", source)
    print(json.dumps(out["counts"], indent=2))
    print("types:", out["types"])
    print("sizes:", out["sizes"])
    print("foils:", out["foils"])
    print("unique entries:", out["counts"]["unique_entries"])


if __name__ == "__main__":
    main()

