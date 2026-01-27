
import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import AsyncStorage from '@react-native-async-storage/async-storage';

export interface Card {
    id: string;
    name: string;
    image: string;
    price?: string;
    timestamp: number;
    language?: string;
    variant?: string;
    condition?: string;
    is_cached?: boolean;
    full_data?: any;
    last_price_sync?: string;
}

interface AppState {
    portfolio: Card[];
    addToPortfolio: (card: Card) => void;
    removeFromPortfolio: (id: string) => void;
    removeOneFromPortfolio: (id: string) => void;
    updateCard: (id: string, updates: Partial<Card>) => void;
    hasOnboarded: boolean;
    setHasOnboarded: (val: boolean) => void;

    // Caching state
    imageStore: Record<string, string>; // imageUrl -> localUri
    marketPrices: Record<string, string>; // cardId -> priceString
    setTopCards: Record<string, string[]>; // setId -> [cardId, ...]
    setTopCardImages: Record<string, string[]>; // setId -> [imagePath, ...]
    updateImageStore: (url: string, uri: string) => void;
    updateMarketPrice: (id: string, price: string) => void;
    updateMarketPrices: (prices: Record<string, string>) => void;
    updateSetTopCards: (setId: string, cardIds: string[], imagePaths: string[]) => void;
}

export const useStore = create<AppState>()(
    persist(
        (set) => ({
            portfolio: [],
            addToPortfolio: (card) => set((state) => ({ portfolio: [card, ...state.portfolio] })),
            removeFromPortfolio: (id) => set((state) => ({ portfolio: state.portfolio.filter((c) => c.id !== id) })),
            removeOneFromPortfolio: (id) => set((state) => {
                const index = state.portfolio.findIndex(c => c.id === id);
                if (index === -1) return state;
                const newPortfolio = [...state.portfolio];
                newPortfolio.splice(index, 1);
                return { portfolio: newPortfolio };
            }),
            updateCard: (id, updates) => set((state) => ({
                portfolio: state.portfolio.map((c) =>
                    c.id === id ? { ...c, ...updates } : c
                )
            })),
            hasOnboarded: false,
            setHasOnboarded: (val) => set({ hasOnboarded: val }),
            imageStore: {},
            updateImageStore: (url, uri) => set((state) => ({
                imageStore: { ...state.imageStore, [url]: uri }
            })),
            marketPrices: {},
            updateMarketPrice: (id, price) => set((state) => ({
                marketPrices: { ...state.marketPrices, [id]: price }
            })),
            updateMarketPrices: (newPrices) => set((state) => ({
                marketPrices: { ...state.marketPrices, ...newPrices }
            })),
            setTopCards: {},
            setTopCardImages: {},
            updateSetTopCards: (setId, cardIds, imagePaths) => set((state) => ({
                setTopCards: { ...state.setTopCards, [setId]: cardIds },
                setTopCardImages: { ...state.setTopCardImages, [setId]: imagePaths }
            })),
        }),
        {
            name: 'poketrader-storage',
            storage: createJSONStorage(() => AsyncStorage),
        }
    )
);
