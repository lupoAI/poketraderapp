
import * as FileSystem from 'expo-file-system/legacy';
import { useStore } from '../store';

const CARDS_DIR = `${FileSystem.documentDirectory}cards/`;
const SETS_DIR = `${FileSystem.documentDirectory}sets/`;
const THUMBS_DIR = `${FileSystem.documentDirectory}thumbs/`;

export async function ensureDirsExist() {
    for (const dir of [CARDS_DIR, SETS_DIR, THUMBS_DIR]) {
        const dirInfo = await FileSystem.getInfoAsync(dir);
        if (!dirInfo.exists) {
            await FileSystem.makeDirectoryAsync(dir, { intermediates: true });
        }
    }
}

export async function cacheCardMetadataAndImage(cardId: string) {
    const store = useStore.getState();
    const card = store.portfolio.find(c => c.id === cardId);

    // Skip if already cached or not in portfolio
    if (!card || card.is_cached) return;

    try {
        await ensureDirsExist();
        const response = await fetch(`https://api.tcgdex.net/v2/en/cards/${cardId}`);
        if (!response.ok) throw new Error('API request failed');
        const data = await response.json();

        if (!data.image) throw new Error('Card image URL missing in metadata');

        const remoteUri = `${data.image}/low.jpg`;
        const localUri = `${CARDS_DIR}${cardId.replace(/[:\/]/g, '_')}_low.jpg`;

        const downloadRes = await FileSystem.downloadAsync(remoteUri, localUri);
        if (downloadRes.status === 200) {
            store.updateCard(cardId, {
                full_data: data,
                is_cached: true
            });
            store.updateImageStore(remoteUri, downloadRes.uri);
        }
    } catch (e) {
        console.log(`[Cache] Skipping card ${cardId}: ${e instanceof Error ? e.message : 'Unknown error'}`);
    }
}

export async function cacheSetLogo(setId: string, logoUrl: string) {
    const store = useStore.getState();
    if (!logoUrl || typeof logoUrl !== 'string') return;

    const remoteUrl = logoUrl.startsWith('http')
        ? (logoUrl.endsWith('.png') ? logoUrl : `${logoUrl}.png`)
        : logoUrl.endsWith('.png') ? logoUrl : `${logoUrl}.png`;

    if (store.imageStore[remoteUrl]) return;

    try {
        await ensureDirsExist();
        const localUri = `${SETS_DIR}${setId.replace(/[:\/]/g, '_')}_low.png`;

        const downloadRes = await FileSystem.downloadAsync(remoteUrl, localUri);
        if (downloadRes.status === 200) {
            store.updateImageStore(remoteUrl, downloadRes.uri);
        }
    } catch (e) {
        console.log(`[Cache] Skipping set logo ${setId}: ${e instanceof Error ? e.message : 'Unknown error'}`);
    }
}

export async function cacheCardThumbnail(cardId: string, imageUrl: string) {
    const store = useStore.getState();
    if (!imageUrl || typeof imageUrl !== 'string') return;

    const remoteUrl = imageUrl.endsWith('/low.jpg') ? imageUrl : `${imageUrl}/low.jpg`;

    if (store.imageStore[remoteUrl]) return;

    try {
        await ensureDirsExist();
        const localUri = `${THUMBS_DIR}${cardId.replace(/[:\/]/g, '_')}_thumb.jpg`;

        const downloadRes = await FileSystem.downloadAsync(remoteUrl, localUri);
        if (downloadRes.status === 200) {
            store.updateImageStore(remoteUrl, downloadRes.uri);
        }
    } catch (e) {
        console.log(`[Cache] Skipping thumbnail ${cardId}: ${e instanceof Error ? e.message : 'Unknown error'}`);
    }
}

export async function clearAllCache() {
    try {
        for (const dir of [CARDS_DIR, SETS_DIR, THUMBS_DIR]) {
            const dirInfo = await FileSystem.getInfoAsync(dir);
            if (dirInfo.exists) {
                await FileSystem.deleteAsync(dir);
            }
        }
    } catch (error) {
        console.error('[Cache] Error clearing cache:', error);
    }
}
