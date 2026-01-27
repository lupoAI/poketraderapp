
import sqlite3
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = REPO_ROOT / "poketrader" / "storage" / "db" / "tcgdex_cards.db"

class DataEngine:
    def __init__(self):
        self.db_path = DB_PATH

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_card_details(self, card_id):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM card_data WHERE card_id = ?", (card_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            d = dict(row)
            # Automatically parse all _json columns
            for key in list(d.keys()):
                if key.endswith("_json") and d[key]:
                    try:
                        # Create a non-json version of the key
                        clean_key = key.replace("_json", "")
                        d[clean_key] = json.loads(d[key])
                    except:
                        pass
            
            # Migration: Rename image_url to image if present (mocking good behavior)
            if "image_url" in d and "image" not in d:
                d["image"] = d.pop("image_url")
                
            return d
        return None

    def get_trending(self):
        # Mocking trends for MVP or simple random selection
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM card_data ORDER BY RANDOM() LIMIT 10")
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]
