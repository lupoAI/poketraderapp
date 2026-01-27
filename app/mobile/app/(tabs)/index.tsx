import { StyleSheet, Image, TouchableOpacity, Platform, View, Text, Modal, ScrollView, Dimensions, useWindowDimensions, Alert } from 'react-native';
import { useEffect, useState, useMemo } from 'react';
import FontAwesome from '@expo/vector-icons/FontAwesome';
import MaterialCommunityIcons from '@expo/vector-icons/MaterialCommunityIcons';
import AsyncStorage from '@react-native-async-storage/async-storage';
import Ionicons from '@expo/vector-icons/Ionicons';
import MaterialIcons from '@expo/vector-icons/MaterialIcons';
import { useStore, Card } from '../../store';
import { router } from 'expo-router';
import { SafeAreaView, useSafeAreaInsets } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { PremiumCard } from '@/components/PremiumCard';
import { InteractiveGraph, GraphInteractionMetrics } from '@/components/InteractiveGraph';
import { API_URL } from '@/constants/Config';
import { getCardImage } from '@/utils/image';
import { getBatchPrices } from '@/utils/tcgdex';
import { generateHistory, formatEuro } from '@/utils/price';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  useAnimatedScrollHandler,
  interpolate,
  Extrapolate
} from 'react-native-reanimated';
import { BlurView } from 'expo-blur';
import { StatusBar } from 'expo-status-bar';
import { cacheCardMetadataAndImage } from '../../utils/cache';
import { SelectionMenu } from '@/components/SelectionMenu';

const { height: SCREEN_HEIGHT } = Dimensions.get('window');

type SortMode = 'value-desc' | 'value-asc' | 'release-desc' | 'added-desc';
type FilterMode = 'all' | 'premium' | 'value' | 'common';

export default function PortfolioScreen() {
  const { portfolio } = useStore();
  const insets = useSafeAreaInsets();
  const { width: screenWidth } = useWindowDimensions();
  const scrollY = useSharedValue(0);

  const [sortMode, setSortMode] = useState<SortMode>('added-desc');
  const [filterMode, setFilterMode] = useState<FilterMode>('all');
  const [showSortMenu, setShowSortMenu] = useState(false);
  const [showFilterMenu, setShowFilterMenu] = useState(false);
  const [deletingCardId, setDeletingCardId] = useState<string | null>(null);

  const { removeFromPortfolio } = useStore();

  const handleDelete = (id: string) => {
    import('expo-haptics').then(Haptics => Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium));
    removeFromPortfolio(id);
    setDeletingCardId(null);
  };

  const handleReset = async () => {
    Alert.alert(
      "Reset App",
      "This will clear all portfolio data and images. Are you sure?",
      [
        { text: "Cancel", style: "cancel" },
        {
          text: "Reset",
          style: "destructive",
          onPress: async () => {
            await AsyncStorage.clear();
            await import('../../utils/cache').then(c => c.clearAllCache());
            // This will effectively wipe the persisted zustand store and reload
            if (Platform.OS === 'web') {
              window.location.reload();
            } else {
              // In Expo Go, the best way to "re-initialize" is to restart or use a state trigger
              // But clearing storage is enough for the next launch.
              Alert.alert("Success", "Data cleared. Please restart the app.");
            }
          }
        }
      ]
    );
  };

  // Graph state
  const [selectedRange, setSelectedRange] = useState<'1W' | '1M' | '3M' | 'All'>('1M');
  const [historicalPrices, setHistoricalPrices] = useState<any[]>([]);
  const [displayPrice, setDisplayPrice] = useState<number>(0);
  const [displayChange, setDisplayChange] = useState<number>(0);
  const [displayAbsChange, setDisplayAbsChange] = useState<number>(0);
  const [displayDate, setDisplayDate] = useState<string>('');
  const [displayIsInteracting, setDisplayIsInteracting] = useState(false);
  const [displayIsPositive, setDisplayIsPositive] = useState(true);

  const parsePrice = (priceStr?: string) => {
    if (!priceStr) return 0;
    const clean = String(priceStr).replace(/[^0-9,.]/g, '');
    return parseFloat(clean.replace(',', '.')) || 0;
  };

  const totalValue = portfolio.reduce((acc, card) => acc + parsePrice(card.price), 0);

  // Initial values for the header when NOT interacting
  const currentHistoryPrice = historicalPrices.length > 0 ? historicalPrices[historicalPrices.length - 1].value : totalValue;
  const initialHistoryPrice = historicalPrices.length > 0 ? historicalPrices[0].value : totalValue;
  const totalChange = currentHistoryPrice - initialHistoryPrice;
  const totalChangePct = initialHistoryPrice !== 0 ? (totalChange / initialHistoryPrice) * 100 : 0;
  const isTotalPositive = totalChange >= 0;

  const handleGraphInteraction = (metrics: GraphInteractionMetrics) => {
    setDisplayPrice(metrics.price);
    setDisplayChange(metrics.change);
    setDisplayAbsChange(metrics.absChange);
    setDisplayIsPositive(metrics.isPositive);
    setDisplayDate(metrics.date);
    setDisplayIsInteracting(metrics.isInteracting);
  };

  const processedData = useMemo(() => {
    // 1. Group duplicates and calculate counts
    const grouped = portfolio.reduce((acc: (Card & { count: number })[], card) => {
      const existing = acc.find(c => c.id === card.id);
      if (existing) {
        existing.count += 1;
        // Keep the most recent timestamp for the grouped item
        if (card.timestamp > existing.timestamp) existing.timestamp = card.timestamp;
      } else {
        acc.push({ ...card, count: 1 });
      }
      return acc;
    }, []);

    // 2. Apply Filtering
    let filtered = grouped;
    if (filterMode === 'premium') {
      filtered = grouped.filter(c => parsePrice(c.price) >= 50);
    } else if (filterMode === 'value') {
      filtered = grouped.filter(c => {
        const p = parsePrice(c.price);
        return p >= 10 && p < 50;
      });
    } else if (filterMode === 'common') {
      filtered = grouped.filter(c => parsePrice(c.price) < 10);
    }

    // 3. Apply Sorting
    return filtered.sort((a, b) => {
      switch (sortMode) {
        case 'value-desc':
          return parsePrice(b.price) - parsePrice(a.price);
        case 'value-asc':
          return parsePrice(a.price) - parsePrice(b.price);
        case 'release-desc':
          const dateA = a.full_data?.set?.releaseDate || '0000-00-00';
          const dateB = b.full_data?.set?.releaseDate || '0000-00-00';
          return dateB.localeCompare(dateA);
        case 'added-desc':
        default:
          return b.timestamp - a.timestamp;
      }
    });
  }, [portfolio, sortMode, filterMode]);

  const scrollHandler = useAnimatedScrollHandler((event) => {
    scrollY.value = event.contentOffset.y;
  });

  // Fetch portfolio history
  useEffect(() => {
    const fetchPortfolioHistory = async () => {
      if (portfolio.length === 0) {
        setHistoricalPrices([]);
        return;
      }

      const rangeMap: any = { '1W': 7, '1M': 30, '3M': 90, 'All': 365 };
      const days = rangeMap[selectedRange];
      const uniqueIds = Array.from(new Set(portfolio.map(c => c.id)));

      try {
        const tcgPrices = await getBatchPrices(uniqueIds);

        // Generate history for each unique card and aggregate
        const aggregate: Record<number, number> = {};

        portfolio.forEach(card => {
          const pricing = tcgPrices[card.id]?.cardmarket;
          const currentPrice = pricing?.avg || pricing?.trend || parsePrice(card.price) || 10;

          const history = generateHistory(currentPrice, days, card.id, {
            avg7: pricing?.avg7,
            avg30: pricing?.avg30
          });

          history.forEach(point => {
            aggregate[point.timestamp] = (aggregate[point.timestamp] || 0) + point.value;
          });
        });

        // Convert to sorted array
        const sortedHistory = Object.entries(aggregate)
          .map(([ts, val]) => ({ timestamp: parseInt(ts), value: val }))
          .sort((a, b) => a.timestamp - b.timestamp);

        setHistoricalPrices(sortedHistory);
      } catch (error) {
        console.error("Failed to fetch portfolio history from TCGdex:", error);
      }
    };

    fetchPortfolioHistory();
  }, [portfolio.length, selectedRange]);

  // Background Cache Cleanup and Sync
  useEffect(() => {
    const syncCache = async () => {
      // 1. Sync images/metadata
      const nonCached = portfolio.filter(c => !c.is_cached);
      if (nonCached.length > 0) {
        for (const card of nonCached) {
          await cacheCardMetadataAndImage(card.id);
        }
      }

      // 2. Sync "live" prices from TCGdex
      const today = new Date().toISOString().split('T')[0];
      const needsUpdate = portfolio.filter(c => c.last_price_sync !== today);

      if (needsUpdate.length > 0) {
        console.log(`[Portfolio] Syncing ${needsUpdate.length} cards with TCGdex...`);
        try {
          const uniqueIds = Array.from(new Set(needsUpdate.map(c => c.id)));
          const prices = await getBatchPrices(uniqueIds);

          // Update store with new prices and sync date
          Object.entries(prices).forEach(([id, pricing]) => {
            const cardmarket = pricing.cardmarket;
            const livePriceValue = cardmarket?.avg || cardmarket?.trend;
            if (livePriceValue) {
              const formattedPrice = formatEuro(livePriceValue);
              console.log(`[Portfolio] Updating ${id} to ${formattedPrice}`);

              // Only trigger update if something actually changed to avoid infinite loops
              const existingCards = portfolio.filter(c => c.id === id);
              const needsLiteralUpdate = existingCards.some(c => c.price !== formattedPrice || c.last_price_sync !== today);

              if (needsLiteralUpdate) {
                useStore.getState().updateCard(id, {
                  price: formattedPrice,
                  last_price_sync: today
                });
              }
            }
          });
        } catch (error) {
          console.error("Failed to sync live prices from TCGdex:", error);
        }
      }
    };
    syncCache();
  }, [portfolio]); // Re-run when portfolio changes, guarded by needsUpdate filter

  const HEADER_MAX_HEIGHT = 380 + insets.top;
  const HEADER_MIN_HEIGHT = 210 + insets.top;
  const SCROLL_DISTANCE = 225;

  const headerStyle = useAnimatedStyle(() => {
    // Reveal the blur and gradient ONLY when scrolling starts
    const opacity = interpolate(
      scrollY.value,
      [0, 60],
      [0, 1],
      Extrapolate.CLAMP
    );
    const borderOpacity = interpolate(
      scrollY.value,
      [140, 180],
      [0, 0.1],
      Extrapolate.CLAMP
    );
    return {
      opacity,
      borderBottomWidth: 1,
      borderBottomColor: `rgba(255,255,255,${borderOpacity})`
    };
  });

  const animatedHeaderContainerStyle = useAnimatedStyle(() => {
    const height = interpolate(
      scrollY.value,
      [0, SCROLL_DISTANCE],
      [HEADER_MAX_HEIGHT, HEADER_MIN_HEIGHT],
      Extrapolate.CLAMP
    );
    return { height };
  });

  const animatedBrandingStyle = useAnimatedStyle(() => {
    // Fixed logo row
    return { transform: [{ translateY: 0 }, { translateX: 0 }, { scale: 1 }] };
  });

  const animatedPriceStyle = useAnimatedStyle(() => {
    const scale = interpolate(scrollY.value, [0, SCROLL_DISTANCE], [1, 0.58], Extrapolate.CLAMP);
    const COMP_W = screenWidth - 48; // Width inside p-6 padding
    // Pure Top-Left Pinning Math:
    // (scale - 1) * (dimension / 2) mathematically anchors the top/left edge.
    const translateX = (scale - 1) * (COMP_W * 0.5);
    const translateY = (scale - 1) * 36; // height ~72 / 2
    return {
      transform: [{ translateX }, { translateY }, { scale }],
      alignItems: 'flex-start',
      width: '100%'
    };
  });

  const animatedMetricsStyle = useAnimatedStyle(() => {
    const mScale = interpolate(scrollY.value, [0, SCROLL_DISTANCE], [1, 0.72], Extrapolate.CLAMP);
    const pScale = interpolate(scrollY.value, [0, SCROLL_DISTANCE], [1, 0.48], Extrapolate.CLAMP);
    const COMP_W = screenWidth - 48;
    const translateX = (mScale - 1) * (COMP_W * 0.5);

    // Follow the bottom of the price + add a bit of gap
    const pHeight = 72;
    const mHeight = 24;
    const pBottomShift = (1 - pScale) * pHeight;
    const mTopComp = (1 - mScale) * (mHeight * 0.5);

    // Add a 6px gap that reveals itself as we scroll up to improve legibility
    const gap = interpolate(scrollY.value, [0, SCROLL_DISTANCE], [0, 6], Extrapolate.CLAMP);
    const translateY = -(pBottomShift + mTopComp) + gap;

    return {
      transform: [{ translateX }, { translateY }, { scale: mScale }],
      alignItems: 'flex-start',
      width: '100%'
    };
  });

  const animatedGraphAreaStyle = useAnimatedStyle(() => {
    const translateY = interpolate(scrollY.value, [0, SCROLL_DISTANCE], [0, -60], Extrapolate.CLAMP);
    const height = interpolate(scrollY.value, [0, SCROLL_DISTANCE], [140, 80], Extrapolate.CLAMP);
    return {
      height,
      transform: [{ translateY }],
      width: screenWidth - 48,
      // Removed overflow: 'hidden' to prevent clipping the chart path
    };
  });

  const animatedRangeStyle = useAnimatedStyle(() => {
    const opacity = interpolate(scrollY.value, [0, 60], [1, 0], Extrapolate.CLAMP);
    const scale = interpolate(scrollY.value, [0, 60], [1, 0.85], Extrapolate.CLAMP);
    const translateY = interpolate(scrollY.value, [0, 60], [0, -10], Extrapolate.CLAMP);
    return { opacity, transform: [{ scale }, { translateY }] };
  });

  const brandingStyle = useAnimatedStyle(() => ({ transform: [{ scale: 1 }] }));


  return (
    <View className="flex-1 bg-black">
      <StatusBar style="light" />

      <SelectionMenu
        visible={showSortMenu}
        onClose={() => setShowSortMenu(false)}
        title="Sort Collection"
        current={sortMode}
        onSelect={setSortMode}
        options={[
          { label: 'Recently Added', value: 'added-desc' },
          { label: 'Value (High to Low)', value: 'value-desc' },
          { label: 'Value (Low to High)', value: 'value-asc' },
          { label: 'Newest First (Sets)', value: 'release-desc' },
        ]}
      />

      <SelectionMenu
        visible={showFilterMenu}
        onClose={() => setShowFilterMenu(false)}
        title="Filter by Value"
        current={filterMode}
        onSelect={setFilterMode}
        options={[
          { label: 'Show All', value: 'all' },
          { label: 'Premium (> 50€)', value: 'premium' },
          { label: 'Value (10€ - 50€)', value: 'value' },
          { label: 'Common (< 10€)', value: 'common' },
        ]}
      />

      {/* Dynamic Header Overlay (Background) - Fades in on scroll */}
      <Animated.View
        pointerEvents="none"
        style={[{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          zIndex: 10,
        }, animatedHeaderContainerStyle, headerStyle]}
      >
        <BlurView intensity={65} tint="dark" style={StyleSheet.absoluteFill} />
        <LinearGradient
          colors={['rgba(0,0,0,0.85)', 'rgba(0,0,0,0.6)', 'rgba(0,0,0,0.3)', 'transparent']}
          locations={[0, 0.4, 0.8, 1]}
          style={StyleSheet.absoluteFill}
        />
      </Animated.View>

      <Animated.FlatList
        data={processedData}
        onScroll={scrollHandler}
        scrollEventThrottle={16}
        keyExtractor={(item) => item.id}
        contentContainerStyle={{
          paddingHorizontal: 8,
          paddingTop: HEADER_MAX_HEIGHT + 10,
          paddingBottom: 160
        }}
        ListHeaderComponent={<View className="h-4" />}
        numColumns={3}
        renderItem={({ item }) => (
          <PremiumCard
            id={item.id}
            name={item.name}
            image={getCardImage(item.image) || undefined}
            price={item.price || "--- €"}
            count={item.count}
            isOwned={true}
            showCheckmark={false}
            isDeleting={deletingCardId === item.id}
            onLongPress={() => {
              import('expo-haptics').then(H => H.impactAsync(H.ImpactFeedbackStyle.Medium));
              setDeletingCardId(item.id);
            }}
            onDelete={() => handleDelete(item.id)}
            onPress={() => {
              if (deletingCardId) {
                setDeletingCardId(null);
              } else {
                router.push(`/card/${item.id}`);
              }
            }}
          />
        )}
      />

      {/* Floating Action Elements (Fixed Top) */}
      <View
        pointerEvents="box-none"
        style={{ position: 'absolute', top: 0, left: 0, right: 0, zIndex: 11 }}
      >
        <SafeAreaView edges={['top']} pointerEvents="box-none">
          <View className="p-6 pt-2" pointerEvents="box-none">
            {/* Persistent Top Row */}
            <Animated.View
              className="flex-row justify-between items-center mb-8"
              pointerEvents="none"
              style={animatedBrandingStyle}
            >
              <View style={[{ flexDirection: 'row', alignItems: 'center' }]}>
                <Text style={{ fontSize: 24, fontWeight: '900', color: 'white', letterSpacing: -1 }}>
                  Poke<Text style={{ color: '#D5A100' }}>Trader</Text>
                </Text>
              </View>
              <View className="flex-row items-center gap-5">
                <TouchableOpacity onPress={() => setShowFilterMenu(true)}>
                  <Ionicons
                    name="options-outline"
                    size={22}
                    color={filterMode !== 'all' ? '#D5A100' : 'white'}
                    style={{ opacity: filterMode !== 'all' ? 1 : 0.6 }}
                  />
                </TouchableOpacity>
                <TouchableOpacity onPress={() => setShowSortMenu(true)}>
                  <Ionicons name="swap-vertical" size={22} color="white" style={{ opacity: 0.6 }} />
                </TouchableOpacity>
                <TouchableOpacity onPress={handleReset}>
                  <Ionicons name="trash-outline" size={22} color="#F87171" style={{ opacity: 0.8 }} />
                </TouchableOpacity>
              </View>
            </Animated.View>

            {/* Price Area: Split into two animated blocks for independent control */}
            <Animated.View className="items-start mb-1" style={animatedPriceStyle} pointerEvents="none">
              <View className="flex-row items-baseline">
                <Text className="text-white text-6xl font-extrabold">€</Text>
                <Text className="text-white text-6xl font-extrabold ml-2">
                  {Math.floor(displayIsInteracting ? displayPrice : totalValue).toLocaleString('de-DE')}
                </Text>
                <Text className="text-white text-2xl font-bold opacity-60">
                  .{((displayIsInteracting ? displayPrice : totalValue) % 1).toFixed(2).split('.')[1]}
                </Text>
              </View>
            </Animated.View>

            <Animated.View className="items-start mb-4" style={animatedMetricsStyle} pointerEvents="none">
              <View className="flex-row items-center flex-wrap">
                <Text className={`${(displayIsInteracting ? displayIsPositive : isTotalPositive) ? 'text-green-400' : 'text-red-400'} font-bold text-base mr-2`}>
                  {(displayIsInteracting ? displayIsPositive : isTotalPositive) ? '+' : '-'}€{Math.abs(displayIsInteracting ? displayAbsChange : totalChange).toFixed(2)}
                </Text>
                <View className={`${(displayIsInteracting ? displayIsPositive : isTotalPositive) ? 'bg-green-500/20 border-green-500/30' : 'bg-red-500/20 border-red-500/30'} px-2 py-0.5 rounded border flex-row items-center mr-2`}>
                  <FontAwesome
                    name={(displayIsInteracting ? displayIsPositive : isTotalPositive) ? "caret-up" : "caret-down"}
                    size={13}
                    color={(displayIsInteracting ? displayIsPositive : isTotalPositive) ? '#4ADE80' : '#F87171'}
                  />
                  <Text className={`${(displayIsInteracting ? displayIsPositive : isTotalPositive) ? 'text-green-400' : 'text-red-400'} font-bold text-sm ml-1`}>
                    {Math.abs(displayIsInteracting ? displayChange : totalChangePct).toFixed(2)}%
                  </Text>
                </View>
                <Text className="text-gray-500 text-sm font-medium">
                  · {displayIsInteracting ? displayDate : `Past ${selectedRange === '1W' ? 'Week' : selectedRange === '1M' ? 'Month' : selectedRange === '3M' ? '3 Months' : 'Year'}`}
                </Text>
              </View>
            </Animated.View>

            {/* Portfolio Chart */}
            <Animated.View className="items-center mb-4" style={animatedGraphAreaStyle}>
              <InteractiveGraph
                data={historicalPrices}
                width={screenWidth - 48}
                height={140}
                accentColor="#D5A100"
                onInteractionUpdate={handleGraphInteraction}
              />
            </Animated.View>

            <Animated.View className="flex-row gap-1 bg-white/5 p-1 rounded-full self-center border border-white/5 mb-8" style={animatedRangeStyle}>
              {(['1W', '1M', '3M', 'All'] as const).map((t) => (
                <TouchableOpacity
                  key={t}
                  onPress={() => setSelectedRange(t)}
                  className={`${selectedRange === t ? 'bg-white/10' : ''} px-5 py-2 rounded-full`}
                >
                  <Text className={`${selectedRange === t ? 'text-white' : 'text-gray-500'} font-bold text-xs`}>{t}</Text>
                </TouchableOpacity>
              ))}
            </Animated.View>
          </View>
        </SafeAreaView>
      </View>

    </View>
  );
}
