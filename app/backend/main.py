
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

# In-memory price cache for daily items
# Format: {card_id: {"price": float, "date": str}}
price_cache = {}

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


class CardResponse(BaseModel):
    card_id: str
    name: Optional[str] = None
    confidence: Optional[float] = None
    image: Optional[str] = None
    price: Optional[str] = None # Simplified

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

@app.get("/api/card/{card_id}")
def get_card(card_id: str):
    data = data_engine.get_card_details(card_id)
    if not data:
        raise HTTPException(status_code=404, detail="Card not found")
    return data


@app.post("/api/portfolio/prices")
def get_portfolio_prices(card_ids: List[str], days: int = 30):
    if not card_ids:
        return []
        
    aggregate_history = {} # timestamp -> total_value
    
    for cid in card_ids:
        # We reuse the deterministic logic from get_card_prices
        seed_str = hashlib.md5(cid.encode()).hexdigest()
        seed = int(seed_str[:8], 16)
        random.seed(seed)
        
        base_price = 5.0 + (seed % 195)
        mu = 0.0002
        sigma = 0.015
        
        current_price = base_price
        now = datetime.now()
        
        for i in range(365):
            epsilon = random.gauss(0, 1)
            current_price = current_price * math.exp((mu - 0.5 * (sigma**2)) + sigma * epsilon)
            
            # Use 1Y history to ensure we can slice any range
            timestamp = int((now - timedelta(days=(364 - i))).timestamp() * 1000)
            
            if timestamp not in aggregate_history:
                aggregate_history[timestamp] = 0.0
            aggregate_history[timestamp] += current_price

    # Convert dict to sorted list of objects
    sorted_history = [
        {"timestamp": ts, "value": round(val, 2)}
        for ts, val in sorted(aggregate_history.items())
    ]
    
    # Slice the requested number of days from the end
    return sorted_history[-days:]

@app.get("/api/card/{card_id}/prices")
def get_card_prices(card_id: str, days: int = 30):
    # Deterministic seed based on card_id
    seed_str = hashlib.md5(card_id.encode()).hexdigest()
    seed = int(seed_str[:8], 16)
    random.seed(seed)
    
    # Base price based on the card_id hash (random value between 5 and 200)
    base_price = 5.0 + (seed % 195)
    
    # Geometric Brownian Motion params
    # We use very small drift and volatility to keep the price realistic over 1Y
    mu = 0.0002  
    sigma = 0.015 
    
    prices = []
    current_price = base_price
    
    # Generate 365 days of data ALWAYS to keep it deterministic across requests
    now = datetime.now()
    full_history = []
    for i in range(365):
        epsilon = random.gauss(0, 1)
        current_price = current_price * math.exp((mu - 0.5 * (sigma**2)) + sigma * epsilon)
        
        timestamp = int((now - timedelta(days=(364 - i))).timestamp() * 1000)
        full_history.append({
            "timestamp": timestamp,
            "value": round(current_price, 2)
        })
    
    # Return only the requested number of days from the end
    return full_history[-days:]

@app.get("/api/card/{card_id}/price")
def get_card_price(card_id: str):
    today = get_today_str()
    
    if card_id in price_cache and price_cache[card_id]["date"] == today:
        return {"card_id": card_id, "price": price_cache[card_id]["price"], "cached": True}
    
    price = generate_card_price(card_id)
    price_cache[card_id] = {"price": price, "date": today}
    
    return {"card_id": card_id, "price": price, "cached": False}

@app.post("/api/cards/prices")
def get_batch_prices(card_ids: List[str]):
    today = get_today_str()
    results = {}
    
    for cid in card_ids:
        if cid in price_cache and price_cache[cid]["date"] == today:
            results[cid] = price_cache[cid]["price"]
        else:
            price = generate_card_price(cid)
            price_cache[cid] = {"price": price, "date": today}
            results[cid] = price
            
    return results

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
