import { StyleSheet, FlatList, View, Text, ActivityIndicator, TextInput, TouchableOpacity, useWindowDimensions, Platform } from 'react-native';
import { useRouter } from 'expo-router';
import { useEffect, useState, useMemo, useRef } from 'react';
import { SafeAreaView, useSafeAreaInsets } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { useStore } from '../../store';
import { SetCard } from '@/components/SetCard';
import FontAwesome from '@expo/vector-icons/FontAwesome';
import Animated, {
    useSharedValue,
    useAnimatedStyle,
    useAnimatedScrollHandler,
    interpolate,
    Extrapolate
} from 'react-native-reanimated';
import { BlurView } from 'expo-blur';
import { cacheSetLogo, cacheCardThumbnail } from '../../utils/cache';
import { StatusBar } from 'expo-status-bar';
import { getBatchPrices } from '@/utils/tcgdex';

export default function SetsScreen() {
    const router = useRouter();
    const insets = useSafeAreaInsets();
    // Stable top inset to prevent layout shift on mount (insets.top starts at 0)
    const topInset = insets.top > 0 ? insets.top : (Platform.OS === 'ios' ? 50 : 30);

    const { portfolio, setTopCards } = useStore();
    const [sets, setSets] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);
    const [search, setSearch] = useState("");
    const scrollY = useSharedValue(0);
    const processingIds = useRef<Set<string>>(new Set());

    const scrollHandler = useAnimatedScrollHandler({
        onScroll: (event) => {
            scrollY.value = event.contentOffset.y;
        },
    });

    useEffect(() => {
        const fetchSets = async () => {
            try {
                const response = await fetch('https://api.tcgdex.net/v2/en/sets');
                if (!response.ok) throw new Error('Failed to fetch sets');
                const data = await response.json();
                setSets(data);

                // Background tasks: logos and thumbnails
                // EXACTLY replicate UI logic: Filter for logos, sort by Owned then Recent
                const withLogos = data.filter((s: any) => s.logo && s.logo !== "");
                const prioritized = withLogos.map((set: any, index: number) => {
                    const ownedInSet = portfolio.filter(c => {
                        const cardSetId = c.id.split('-')[0];
                        return cardSetId === set.id || c.full_data?.set?.id === set.id;
                    });
                    const uniqueOwned = new Set(ownedInSet.map(c => c.id)).size;
                    return {
                        ...set,
                        originalIndex: index, // Preserve index from original 'data' array
                        ownedCount: uniqueOwned,
                        percentage: set.cardCount?.total > 0 ? (uniqueOwned / set.cardCount.total) * 100 : 0
                    };
                }).sort((a: any, b: any) => {
                    // 1. Owned sets first
                    if (b.percentage !== a.percentage) return b.percentage - a.percentage;
                    // 2. Most recent sets (highest original index) second
                    return b.originalIndex - a.originalIndex;
                });

                // Cache logos for all prioritized sets (up to 120, small assets)
                const logoPriority = prioritized.slice(0, 120);
                logoPriority.forEach((s: any) => {
                    if (s.logo) cacheSetLogo(s.id, s.logo);
                });

                // NOTE: We no longer pre-fetch all podiums. We lazy load them via onViewableItemsChanged.
                // This ensures we only fetch for sets the user actually sees.

            } catch (err) {
                console.error(err);
            } finally {
                setLoading(false);
            }
        };
        fetchSets();
    }, []);

    const viewabilityConfig = useRef({
        itemVisiblePercentThreshold: 10,
        waitForInteraction: false,
        minimumViewTime: 300,
    }).current;

    // Set Podium Fetcher
    const fetchPodium = async (setId: string) => {
        // Prevent duplicate fetches if already have data or currently processing
        const currentData = useStore.getState().setTopCards[setId];
        if (currentData && currentData.length > 0) return;
        if (processingIds.current.has(setId)) return;

        processingIds.current.add(setId);

        try {
            const res = await fetch(`https://api.tcgdex.net/v2/en/sets/${setId}`);
            if (!res.ok) throw new Error("Set fetch failed");
            const detailedSet = await res.json();

            if (detailedSet.cards) {
                // HEURISTIC: Find potentially valuable cards
                const allCards = detailedSet.cards;
                // Last 15 (Secrets/Ultras) + First 3 (Old Holos)
                const last15 = allCards.slice(-15);
                const first3 = allCards.slice(0, 3);
                const candidateCards = [...new Set([...first3, ...last15])];

                // Fetch prices
                const candidateIds = candidateCards.map((c: any) => c.id);
                const prices = await getBatchPrices(candidateIds);

                // Sort by price
                const sortedCandidates = candidateCards.sort((a: any, b: any) => {
                    const pA = prices[a.id]?.cardmarket;
                    const pB = prices[b.id]?.cardmarket;
                    const valA = pA?.avg || pA?.trend || 0;
                    const valB = pB?.avg || pB?.trend || 0;
                    return valB - valA;
                });

                // Update Store with Top 3
                const top3 = sortedCandidates.slice(0, 3);
                const topIds = top3.map((c: any) => c.id);
                const topImages = top3.map((c: any) => c.image);

                // Only update if we found cards (prevents infinite refetching of empty sets)
                if (topIds.length > 0) {
                    useStore.getState().updateSetTopCards(setId, topIds, topImages);
                    // Cache thumbnails
                    for (const c of top3) {
                        await cacheCardThumbnail(c.id, c.image);
                    }
                }
            }
        } catch (e) {
            console.log(`[Podium] Failed for ${setId}`, e);
            // Remove from processing so we can try again if user scrolls back
            processingIds.current.delete(setId);
        }
    };

    const onViewableItemsChanged = useMemo(() => {
        return ({ viewableItems }: { viewableItems: any[] }) => {
            viewableItems.forEach(item => {
                if (item.isViewable) {
                    // Trigger fetch for viewable sets
                    // We use a slight delay or debounce could be added, but minimal for now
                    fetchPodium(item.item.id);
                }
            });
        };
    }, []);

    const processedSets = useMemo(() => {
        if (!sets.length) return [];
        const setStats = sets.map((set, index) => {
            const ownedInSet = portfolio.filter(c => {
                const cardSetId = c.id.split('-')[0];
                return cardSetId === set.id || c.full_data?.set?.id === set.id;
            });
            const uniqueOwned = new Set(ownedInSet.map(c => c.id)).size;
            return {
                ...set,
                originalIndex: index,
                ownedCount: uniqueOwned,
                percentage: set.cardCount?.total > 0 ? (uniqueOwned / set.cardCount.total) * 100 : 0
            };
        });
        const withLogo = setStats.filter(s => s.logo && s.logo !== "");
        const filtered = search
            ? withLogo.filter(s => s.name.toLowerCase().includes(search.toLowerCase()))
            : withLogo;

        return filtered.sort((a, b) => {
            if (b.percentage !== a.percentage) return b.percentage - a.percentage;
            return b.originalIndex - a.originalIndex;
        });
    }, [sets, portfolio, search]);

    const headerBlurStyle = useAnimatedStyle(() => {
        const opacity = interpolate(scrollY.value, [0, 60], [0, 1], Extrapolate.CLAMP);
        return { opacity };
    });

    const searchBarStyle = useAnimatedStyle(() => {
        const opacity = interpolate(scrollY.value, [0, 50], [1, 0], Extrapolate.CLAMP);
        const translateY = interpolate(scrollY.value, [0, 80], [0, -60], Extrapolate.CLAMP);
        const scale = interpolate(scrollY.value, [0, 50], [1, 0.95], Extrapolate.CLAMP);
        return {
            opacity,
            transform: [{ translateY }, { scale }],
            pointerEvents: scrollY.value > 40 ? 'none' : 'auto'
        };
    });

    return (
        <View className="flex-1 bg-black">
            <StatusBar style="light" />
            <LinearGradient
                colors={['rgba(213, 161, 0, 0.15)', 'rgba(0,0,0,0)']}
                style={StyleSheet.absoluteFill}
            />

            {/* Dynamic Header Overlay */}
            <Animated.View
                style={[{
                    position: 'absolute', top: 0, left: 0, right: 0, zIndex: 10,
                    height: 90 + topInset,
                }, headerBlurStyle]}
            >
                <BlurView intensity={40} tint="dark" style={StyleSheet.absoluteFill} />
                <LinearGradient
                    colors={['rgba(0,0,0,1)', 'rgba(0,0,0,0.8)', 'transparent']}
                    locations={[0, 0.6, 1]}
                    style={StyleSheet.absoluteFill}
                />
            </Animated.View>

            {/* Static Content Header - Titles and Search */}
            <View style={{ position: 'absolute', top: 0, left: 0, right: 0, zIndex: 11, paddingTop: topInset }}>
                <View className="px-6 py-4">
                    <View className="mb-2">
                        <Text className="text-4xl font-bold text-white">Sets</Text>
                    </View>

                    <Animated.View style={searchBarStyle}>
                        <View className="bg-white/10 flex-row items-center px-6 py-4 rounded-3xl border border-white/10 backdrop-blur-xl mb-2 mt-4">
                            <FontAwesome name="search" size={16} color="#888" />
                            <TextInput
                                className="flex-1 ml-3 text-white font-semibold"
                                placeholder="Search sets..."
                                placeholderTextColor="#555"
                                value={search}
                                onChangeText={setSearch}
                            />
                        </View>
                    </Animated.View>
                </View>
            </View>

            {loading ? (
                <View className="flex-1 items-center justify-center pt-20">
                    <ActivityIndicator color="#D5A100" size="large" />
                </View>
            ) : (
                <Animated.FlatList
                    data={processedSets}
                    onScroll={scrollHandler}
                    scrollEventThrottle={16}
                    keyExtractor={(item) => item.id}
                    numColumns={2}
                    onViewableItemsChanged={onViewableItemsChanged}
                    viewabilityConfig={viewabilityConfig}
                    contentContainerStyle={{
                        paddingHorizontal: 8,
                        paddingTop: 160 + topInset,
                        paddingBottom: 160
                    }}
                    renderItem={({ item }) => (
                        <SetCard
                            id={item.id}
                            name={item.name}
                            logo={item.logo}
                            topCards={setTopCards[item.id] || []}
                            topCardImages={useStore.getState().setTopCardImages[item.id] || []}
                            ownedCount={item.ownedCount}
                            totalCount={item.cardCount?.total || 0}
                            onPress={() => router.push({
                                pathname: `/set/${item.id}`,
                                params: { name: item.name, logo: item.logo }
                            } as any)}
                            onCardPress={(cardId, image) => router.push({
                                pathname: `/card/${cardId}`,
                                params: { placeholder: image }
                            } as any)}
                        />
                    )}
                />
            )}
        </View>
    );
}
