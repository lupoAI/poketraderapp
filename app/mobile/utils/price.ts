
/**
 * Utility for generating deterministic pseudo-random price history
 * for Pokemon cards, anchored by real TCGdex pricing data.
 */

interface DailyPrice {
    timestamp: number;
    value: number;
}

/**
 * Geometric Brownian Motion generator for realistic looking financial charts.
 * 
 * @param currentPrice The "end" price point (most recent)
 * @param days Number of days to generate back in history
 * @param seed Seed for deterministic randomness (usually cardId)
 * @param anchors Optional anchor points to guide the generation (e.g. avg7, avg30)
 */
export function generateHistory(
    currentPrice: number,
    days: number = 30,
    seed: string = 'default',
    anchors?: { avg7?: number; avg30?: number }
): DailyPrice[] {
    // Simple hash for seed
    let hash = 0;
    for (let i = 0; i < seed.length; i++) {
        hash = ((hash << 5) - hash) + seed.charCodeAt(i);
        hash |= 0; // Convert to 32bit integer
    }

    // Linear feedback shift register for deterministic "random"
    const random = () => {
        hash = (hash * 1664525 + 1013904223) | 0;
        return (hash >>> 0) / 0xffffffff;
    };

    const normalRandom = () => {
        // Box-Muller transform
        const u = 1 - random();
        const v = 1 - random();
        return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    };

    const history: DailyPrice[] = [];
    const now = new Date();
    now.setHours(0, 0, 0, 0);
    const startOfDay = now.getTime();
    const MS_PER_DAY = 24 * 60 * 60 * 1000;

    // Parameters for Geometric Brownian Motion
    const mu = 0.0002; // drift
    const sigma = 0.015; // volatility (1.5% daily)

    let price = currentPrice;

    // We generate backwards from today
    for (let i = 0; i < days; i++) {
        const timestamp = startOfDay - (i * MS_PER_DAY);

        // Add point to start of array
        history.unshift({
            timestamp,
            value: Math.max(0.01, Math.round(price * 100) / 100)
        });

        // Walk backwards
        // dS = S * (mu*dt + sigma*dW)
        // Here dt = 1 (one day)
        // Since we go backwards, we invert the drift and add noise
        const epsilon = normalRandom();
        const drift = mu;
        const diffusion = sigma * epsilon;

        // To go backwards realistically: S_prev = S_next / exp(drift + diffusion)
        price = price / Math.exp(drift + diffusion);

        // Apply soft anchors if they exist
        // This nudges the walk towards the known averages if we have them
        if (i === 7 && anchors?.avg7) {
            // If our current walk at day 7 is far from avg7, nudge it halfway
            price = (price + anchors.avg7) / 2;
        }
        if (i === 30 && anchors?.avg30) {
            price = (price + anchors.avg30) / 2;
        }
    }

    return history;
}

/**
 * Format a number as Euro currency.
 */
export function formatEuro(value: number | string | undefined): string {
    if (value === undefined || value === null) return '€ ---';
    const num = typeof value === 'string' ? parseFloat(value.replace(/[^0-9,.]/g, '').replace(',', '.')) : value;
    if (isNaN(num)) return '€ ---';
    return `€${num.toFixed(2)}`;
}
