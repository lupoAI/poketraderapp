
const TCGDEX_API_URL = 'https://api.tcgdex.net/v2/en';
const TCGDEX_GRAPHQL_URL = 'https://api.tcgdex.net/v2/graphql';

export interface TCGdexPriceData {
  updated: string;
  unit: string;
  avg?: number;
  low?: number;
  trend?: number;
  avg1?: number;
  avg7?: number;
  avg30?: number;
  'avg-holo'?: number;
  'low-holo'?: number;
  'trend-holo'?: number;
  'avg1-holo'?: number;
  'avg7-holo'?: number;
  'avg30-holo'?: number;
}

export interface TCGdexCardPricing {
  cardmarket?: TCGdexPriceData;
  tcgplayer?: {
    updated: string;
    unit: string;
    normal?: {
      marketPrice?: number;
      lowPrice?: number;
    };
    holofoil?: {
      marketPrice?: number;
      lowPrice?: number;
    };
  };
}

/**
 * Fetches batch pricing for cards using TCGdex REST API.
 * Since GraphQL currently lacks 'pricing' field, we fetch individual cards in parallel.
 */
export async function getBatchPrices(cardIds: string[]): Promise<Record<string, TCGdexCardPricing>> {
  if (cardIds.length === 0) return {};

  const uniqueIds = Array.from(new Set(cardIds));
  console.log(`[TCGdex] Fetching prices for ${uniqueIds.length} cards via REST...`);

  const prices: Record<string, TCGdexCardPricing> = {};

  // We fetch in small chunks to avoid rate limits or overwhelming the network
  // TCGdex handles ~50/sec easily, so 25 parallel requests is safe and fast
  const CHUNK_SIZE = 25;
  for (let i = 0; i < uniqueIds.length; i += CHUNK_SIZE) {
    const chunk = uniqueIds.slice(i, i + CHUNK_SIZE);
    const results = await Promise.all(
      chunk.map(async (id) => {
        try {
          const response = await fetch(`${TCGDEX_API_URL}/cards/${id}`);
          if (!response.ok) return null;
          const data = await response.json();
          return { id, pricing: data.pricing };
        } catch (e) {
          console.error(`[TCGdex] Failed to fetch price for ${id}:`, e);
          return null;
        }
      })
    );

    results.forEach(res => {
      if (res && res.pricing) {
        prices[res.id] = res.pricing;
      }
    });
  }

  console.log(`[TCGdex] Found pricing for ${Object.keys(prices).length}/${uniqueIds.length} requested cards.`);
  return prices;
}

/**
 * Fetches full details for a single card including pricing.
 */
export async function getCardDetails(cardId: string) {
  try {
    const response = await fetch(`${TCGDEX_API_URL}/cards/${cardId}`);
    if (!response.ok) throw new Error(`HTTP error: ${response.status}`);
    return await response.json();
  } catch (error) {
    console.error(`Error fetching card ${cardId}:`, error);
    throw error;
  }
}
