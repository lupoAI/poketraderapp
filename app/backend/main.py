
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import math
import random
import hashlib
import time
from datetime import datetime, timedelta

from ai_engine import AIEngine
from data_engine import DataEngine

REPO_ROOT = Path(__file__).resolve().parents[2]
STORAGE_DIR = REPO_ROOT / "research" / "storage"

app = FastAPI()

# CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Initializing AI Engine...")
ai_engine = AIEngine()
print("Initializing Data Engine...")
data_engine = DataEngine()

# In-memory price cache for daily items is no longer needed
# as pricing is handled on-the-fly or client-side.

def get_today_str():
    return datetime.now().strftime("%Y-%m-%d")

def generate_card_price(card_id: str) -> float:
    # Deterministic price based on card_id and date
    today = get_today_str()
    seed_str = f"{card_id}_{today}"
    seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
    random.seed(seed)
    
    # Base seed for "card quality"
    base_seed_str = hashlib.md5(card_id.encode()).hexdigest()
    base_seed = int(base_seed_str[:8], 16)
    
    # Prices range from 1.0 to 300.0 based on card_id
    base_val = 1.0 + (base_seed % 299)
    
    # Daily fluctuation +/- 5%
    fluctuation = random.uniform(0.95, 1.05)
    return round(base_val * fluctuation, 2)


# CardResponse model removed (unused)

@app.get("/")
def read_root():
    return {"status": "ok", "app": "Poketrader Backend"}

@app.post("/api/identify")
async def identify_card(file: UploadFile = File(...), is_cropped: bool = False):
    # Save image for debugging
    import time
    debug_dir = STORAGE_DIR / "debug_uploads"
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    contents = await file.read()
    
    # Save with timestamp
    timestamp = int(time.time() * 1000)
    with open(debug_dir / f"upload_{timestamp}.jpg", "wb") as f:
        f.write(contents)
        
    matches = ai_engine.detect_and_identify(contents, is_cropped=is_cropped)
    
    if not matches:
        return {"match": False, "detail": "No card detected"}
    
    results = []
    for match in matches:
        enriched_top = []
        for m in match["top_matches"]:
            # Enrich each top match with DB data
            card_data = data_engine.get_card_details(m["card_id"])
            name = m["card_id"]
            # Get deterministic price
            raw_price = generate_card_price(m["card_id"])
            price = f"€{raw_price:.2f}"
            
            if card_data:
                name = card_data.get("name", name)

            # Get deterministic change for "Market Price" badge
            seed_str = f"{m['card_id']}_change_{get_today_str()}"
            seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
            random.seed(seed)
            pct_change = random.uniform(-10.0, 10.0)
            change_str = f"{'+' if pct_change >= 0 else ''}{pct_change:.1f}%"

            enriched_top.append({
                "card_id": m["card_id"],
                "name": name,
                "confidence": m["confidence"],
                "price": price,
                "change": change_str,
                "is_positive": pct_change >= 0,
                "image": card_data.get("image") or card_data.get("image_url"),
                "db_data": card_data
            })

        results.append({
            "center": match["center"],
            "top_matches": enriched_top,
            # Current selection defaults to top index
            "selection_index": 0
        })

    return {
        "match": True,
        "count": len(results),
        "cards": results
    }

@app.get("/api/trends")
def get_trends(range_type: str = Query("1W", alias="range")):
    # Real card ids from Database
    pool = data_engine.get_trending()
    
    # Range configuration
    range_map = {
        "1W": {"days": 7, "volatility": 0.05},
        "1M": {"days": 30, "volatility": 0.08},
        "3M": {"days": 90, "volatility": 0.12},
        "1Y": {"days": 365, "volatility": 0.20}
    }
    
    conf = range_map.get(range_type, range_map["1W"])
    days = conf["days"]
    volatility = conf["volatility"]

    movers = []
    for card in pool:
        card_id = card["card_id"]
        name = card["name"]
        
        # Deterministic generation based on card_id and range
        seed_str = f"{card_id}_{range_type}"
        seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
        random.seed(seed)
        
        # Base price
        base_seed = int(hashlib.md5(card_id.encode()).hexdigest()[:8], 16)
        base_price = 5.0 + (base_seed % 495)
        
        # Generate trend
        current_price = base_price
        history = []
        now = datetime.now()
        
        # We generate a small set of points for the sparkline (e.g., 10 points)
        points = 10
        interval = timedelta(days=days) / points
        
        for i in range(points + 1):
            # Price fluctuation
            change = 1 + (random.uniform(-1, 1) * volatility / points * (i if i > 0 else 1))
            current_price = current_price * change
            
            ts = int((now - (timedelta(days=days) - (interval * i))).timestamp() * 1000)
            history.append({"timestamp": ts, "value": round(current_price, 2)})
            
        start_val = history[0]["value"]
        end_val = history[-1]["value"]
        abs_change = end_val - start_val
        pct_change = (abs_change / start_val) * 100

        image = card.get("image") or card.get("image_url")

        movers.append({
            "card_id": card_id,
            "name": name,
            "price": f"€{end_val:.2f}",
            "change": f"{'+' if pct_change >= 0 else ''}{pct_change:.1f}%",
            "changePct": pct_change,
            "image": image,
            "chartData": history
        })
        
    # Sort and split
    movers.sort(key=lambda x: x["changePct"], reverse=True)
    
    gainers = [m for m in movers if m["changePct"] > 0][:5]
    losers = [m for m in movers if m["changePct"] < 0]
    losers.sort(key=lambda x: x["changePct"]) # Most negative first
    losers = losers[:5]
    
    return {
        "gainers": gainers,
        "losers": losers
    }
