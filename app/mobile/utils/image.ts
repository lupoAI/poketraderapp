import { API_URL } from '@/constants/Config';
import { useStore } from '../store';

/**
 * Standardizes a path into a full remote URL.
 * Strictly relies on path; no fallback ID-based TCGdex lookups.
 */
function resolveRemoteUrl(
    path: string | undefined | null,
    type: 'card' | 'logo',
    quality: 'low' | 'high' = 'low',
    ext: string
): string | undefined {
    if (!path) return undefined;

    if (path.startsWith('file://')) return path;

    if (path.startsWith('http')) {
        // Handle TCGdex direct path consistency
        if (path.includes('tcgdex.net') && !path.endsWith('.jpg') && !path.endsWith('.png')) {
            // Standardize set logos to .png (no /low quality for logos)
            if (type === 'logo') return `${path}.${ext}`;
            return `${path}/${quality}.${ext}`;
        }
        return path;
    }

    // Relative/Identifier fallbacks - Assuming TCGdex if not starting with /
    const base = 'https://api.tcgdex.net/v2';
    if (type === 'logo') return `${base}/${path}.${ext}`;
    return `${base}/${path}/${quality}.${ext}`;
}

/**
 * Resolves a card image URL. 
 * Only checks cache if quality is 'low' and extension is 'jpg'.
 */
export function getCardImage(
    path: string | undefined | null,
    quality: 'low' | 'high' = 'low',
    ext: string = 'jpg'
): string | undefined {
    if (path?.startsWith('file://')) return path;
    const remoteUrl = resolveRemoteUrl(path, 'card', quality, ext);
    if (!remoteUrl) return undefined;
    const { imageStore } = useStore.getState();

    // Cache lookup only for low.jpg as per user request
    if (quality === 'low' && ext === 'jpg' && imageStore[remoteUrl]) {
        return imageStore[remoteUrl];
    }

    return remoteUrl;
}

/**
 * Resolves a set logo URL.
 * Only checks cache if quality is 'low' and extension is 'png'.
 */
export function getLogoImage(
    path: string | undefined | null,
    quality: 'low' | 'high' = 'low',
    ext: string = 'png'
): string | undefined {
    if (path?.startsWith('file://')) return path;
    const remoteUrl = resolveRemoteUrl(path, 'logo', quality, ext);
    if (!remoteUrl) return undefined;
    const { imageStore } = useStore.getState();

    // Cache lookup for logos (using png ext)
    if (ext === 'png' && imageStore[remoteUrl]) {
        return imageStore[remoteUrl];
    }

    return remoteUrl;
}
