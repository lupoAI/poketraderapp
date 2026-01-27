
import { useLocalSearchParams, useRouter, Stack } from 'expo-router';
import { View, Text, StyleSheet, ActivityIndicator, TouchableOpacity, Image, Platform } from 'react-native';
import { SafeAreaView, useSafeAreaInsets } from 'react-native-safe-area-context';
import FontAwesome from '@expo/vector-icons/FontAwesome';
import { useState, useEffect, useMemo } from 'react';
import { LinearGradient } from 'expo-linear-gradient';
import { useStore } from '../../store';
import { PremiumCard } from '@/components/PremiumCard';
import { API_URL } from '@/constants/Config';
import Animated, {
    useSharedValue,
    useAnimatedStyle,
    useAnimatedScrollHandler,
    interpolate,
    Extrapolate
} from 'react-native-reanimated';
import { BlurView } from 'expo-blur';
import { StatusBar } from 'expo-status-bar';
import { cacheCardThumbnail } from '../../utils/cache';
import { SelectionMenu } from '@/components/SelectionMenu';
import { getCardImage, getLogoImage } from '@/utils/image';
import { getBatchPrices } from '@/utils/tcgdex';
import { formatEuro } from '@/utils/price';
import Ionicons from '@expo/vector-icons/Ionicons';

export default function SetDetailScreen() {
    const { id, name: paramName, logo: paramLogo } = useLocalSearchParams() as { id: string, name?: string, logo?: string };
    const router = useRouter();
    const insets = useSafeAreaInsets();
    // Stable top inset to prevent layout shift on mount (insets.top starts at 0)
    const topInset = insets.top > 0 ? insets.top : (Platform.OS === 'ios' ? 50 : 30);

    const { portfolio, marketPrices, updateMarketPrice } = useStore();
    const [setData, setSetData] = useState<any>(null);
    const [loading, setLoading] = useState(true);
    const [sortMode, setSortMode] = useState<'number' | 'value'>('number');
    const [showSortMenu, setShowSortMenu] = useState(false);
    const [highResLogoLoaded, setHighResLogoLoaded] = useState(false);
    const scrollY = useSharedValue(0);

    const scrollHandler = useAnimatedScrollHandler((event) => {
        scrollY.value = event.contentOffset.y;
    });

    useEffect(() => {
        const fetchSetDetails = async () => {
            try {
                const response = await fetch(`https://api.tcgdex.net/v2/en/sets/${id}`);
                if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                const data = await response.json();
                setSetData(data);

                // Start background tasks
                if (data.cards) {
                    // 1. Sync prices using TCGdex REST (Parallel)
                    const cardIds = data.cards.map((c: any) => c.id);
                    getBatchPrices(cardIds).then(prices => {
                        const updates: Record<string, string> = {};
                        Object.entries(prices).forEach(([cid, pricing]) => {
                            const cardmarket = pricing.cardmarket;
                            const livePriceValue = cardmarket?.avg || cardmarket?.trend;
                            if (livePriceValue) {
                                updates[cid] = formatEuro(livePriceValue);
                            }
                        });
                        if (Object.keys(updates).length > 0) {
                            useStore.getState().updateMarketPrices(updates);
                        }
                    }).catch(err => console.error("TCGdex Price sync error:", err));

                    // 2. Cache first 9 thumbnails
                    data.cards.slice(0, 9).forEach((c: any) => {
                        cacheCardThumbnail(c.id, c.image);
                    });
                }
            } catch (error) {
                console.error("Error fetching set details:", error);
            } finally {
                setLoading(false);
            }
        };

        if (id) {
            fetchSetDetails();
        }
    }, [id]);

    const headerStyle = useAnimatedStyle(() => {
        const opacity = interpolate(scrollY.value, [0, 50], [0, 1], Extrapolate.CLAMP);
        return { opacity };
    });

    const brandingStyle = useAnimatedStyle(() => {
        const scale = interpolate(scrollY.value, [0, 50], [1, 0.9], Extrapolate.CLAMP);
        return { transform: [{ scale }] };
    });

    const ownedCards = useMemo(() => {
        return portfolio.filter(c => {
            const cardSetId = c.id.split('-')[0];
            return cardSetId === id || c.full_data?.set?.id === id;
        });
    }, [portfolio, id]);

    const ownedIds = new Set(ownedCards.map(c => c.id));
    const totalInSet = setData?.cardCount?.total || setData?.cards?.length || 0;
    const completionPercentage = totalInSet > 0 ? (ownedIds.size / totalInSet) * 100 : 0;

    const parsePrice = (priceStr?: string) => {
        if (!priceStr) return 0;
        const clean = String(priceStr).replace(/[^0-9,.]/g, '');
        return parseFloat(clean.replace(',', '.')) || 0;
    };

    const sortedCards = useMemo(() => {
        if (!setData?.cards) return [];
        return [...setData.cards].sort((a, b) => {
            if (sortMode === 'value') {
                const priceA = parsePrice(marketPrices[a.id]);
                const priceB = parsePrice(marketPrices[b.id]);
                return priceB - priceA;
            } else {
                return parseInt(a.localId) - parseInt(b.localId);
            }
        });
    }, [setData?.cards, sortMode, marketPrices]);

    return (
        <View className="flex-1 bg-black">
            <StatusBar style="light" />
            <Stack.Screen options={{ headerShown: false }} />

            <SelectionMenu
                visible={showSortMenu}
                onClose={() => setShowSortMenu(false)}
                title="Sort Cards"
                current={sortMode}
                onSelect={setSortMode}
                options={[
                    { label: 'Card Number', value: 'number' },
                    { label: 'Market Value', value: 'value' },
                ]}
            />

            {/* Persistent Header Background */}
            <Animated.View
                style={[{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    right: 0,
                    zIndex: 10,
                    height: 260 + topInset,
                    pointerEvents: 'none'
                }, headerStyle]}
            >
                <BlurView intensity={40} tint="dark" style={StyleSheet.absoluteFill} />
                <LinearGradient
                    colors={['rgba(0,0,0,1)', 'rgba(0,0,0,0.95)', 'rgba(0,0,0,0.8)', 'rgba(0,0,0,0.4)', 'transparent']}
                    locations={[0, 0.4, 0.6, 0.8, 1]}
                    style={StyleSheet.absoluteFill}
                />
            </Animated.View>

            {loading ? (
                <View className="flex-1 items-center justify-center pt-20">
                    <ActivityIndicator color="#D5A100" size="large" />
                </View>
            ) : (
                <Animated.FlatList
                    data={sortedCards}
                    onScroll={scrollHandler}
                    scrollEventThrottle={16}
                    keyExtractor={(item) => item.id}
                    numColumns={3}
                    contentContainerStyle={{
                        paddingHorizontal: 8,
                        paddingTop: 300 + topInset,
                        paddingBottom: 100
                    }}
                    renderItem={({ item }) => {
                        const isOwned = ownedIds.has(item.id);
                        const portfolioInstances = portfolio.filter(c => c.id === item.id);
                        const portfolioCard = portfolioInstances[0];
                        const livePrice = marketPrices[item.id] || portfolioCard?.price || "€ ---";

                        return (
                            <PremiumCard
                                id={item.id}
                                name={item.name}
                                image={getCardImage(item.image)}
                                price={livePrice}
                                count={portfolioInstances.length}
                                isOwned={isOwned}
                                onPress={() => router.push({
                                    pathname: `/card/${item.id}`,
                                    params: { placeholder: item.image }
                                } as any)}
                            />
                        );
                    }}
                />
            )}

            {/* Content Header - INSTANTLY populated via params */}
            <View pointerEvents="box-none" style={{ position: 'absolute', top: 0, left: 0, right: 0, zIndex: 11, paddingTop: topInset }}>
                <View className="p-6 pt-2">
                    <View className="flex-row justify-between items-center mb-8">
                        <TouchableOpacity
                            onPress={() => router.back()}
                            className="w-10 h-10 bg-white/10 rounded-full items-center justify-center border border-white/5"
                            style={{ paddingRight: 2 }}
                        >
                            <Ionicons name="chevron-back" size={24} color="white" />
                        </TouchableOpacity>
                        <Animated.View style={[brandingStyle]}>
                            <Text style={{ fontSize: 18, fontWeight: '900', color: 'white' }} numberOfLines={1}>
                                {setData?.name || paramName}
                            </Text>
                        </Animated.View>
                        <TouchableOpacity onPress={() => setShowSortMenu(true)} className="w-10 h-10 bg-white/10 rounded-full items-center justify-center border border-white/5">
                            <Ionicons name="swap-vertical" size={16} color="white" />
                        </TouchableOpacity>
                    </View>

                    <View className="items-center mb-6">
                        <View className="h-20 w-40 mb-4 items-center justify-center">
                            {(setData?.logo || paramLogo) ? (
                                <View className="w-full h-full">
                                    {getLogoImage(setData?.logo || paramLogo) && (
                                        <Image
                                            source={{ uri: getLogoImage(setData?.logo || paramLogo)! }}
                                            className="w-full h-full"
                                            resizeMode="contain"
                                        />
                                    )}
                                </View>
                            ) : (
                                <View className="w-full h-full bg-white/5 rounded-lg animate-pulse" />
                            )}
                        </View>
                        <View className="h-[64px] items-center justify-center">
                            {loading ? (
                                <View className="items-center">
                                    <ActivityIndicator color="#D5A100" size="small" />
                                    <Text className="text-gray-500 text-[10px] font-bold tracking-[2px] uppercase mt-2">
                                        Loading Set Data...
                                    </Text>
                                </View>
                            ) : (
                                <View className="items-center">
                                    <Text className="text-white text-3xl font-bold mb-1">{ownedIds.size}/{totalInSet}</Text>
                                    <Text className="text-gray-500 text-[10px] font-bold tracking-[2px] uppercase">
                                        Collection Progress • {completionPercentage.toFixed(1)}%
                                    </Text>
                                </View>
                            )}
                        </View>
                    </View>

                    <View className="h-1 bg-white/5 rounded-full overflow-hidden mx-12">
                        <LinearGradient
                            colors={['#FFCB05', '#D5A100']}
                            start={{ x: 0, y: 0 }}
                            end={{ x: 1, y: 0 }}
                            style={{ width: `${completionPercentage}%`, height: '100%' }}
                        />
                    </View>
                </View>
            </View>
        </View>
    );
}
