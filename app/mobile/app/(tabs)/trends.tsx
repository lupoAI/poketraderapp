import { StyleSheet, FlatList, Platform, TouchableOpacity, Dimensions, Image, View, Text, ScrollView, ActivityIndicator } from 'react-native';
import { router } from 'expo-router';
import { useEffect, useState, useMemo } from 'react';
import FontAwesome from '@expo/vector-icons/FontAwesome';
import Ionicons from '@expo/vector-icons/Ionicons';
import MaterialIcons from '@expo/vector-icons/MaterialIcons';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { API_URL } from '@/constants/Config';
import Colors from '@/constants/Colors';
import Animated, { FadeInDown, FadeInRight, Layout } from 'react-native-reanimated';
import { InteractiveGraph, GraphInteractionMetrics } from '@/components/InteractiveGraph';
import { getCardImage } from '@/utils/image';
import { getBatchPrices } from '@/utils/tcgdex';
import { generateHistory, formatEuro } from '@/utils/price';

const { width } = Dimensions.get('window');

type TimeRange = '1W' | '1M' | '3M' | '1Y';

interface MoverData {
    card_id: string;
    name: string;
    price: string;
    change: string;
    changePct: number;
    image?: string;
    chartData: { timestamp: number, value: number }[];
}
const TrendingCard = ({ item, isGainer }: { item: MoverData, isGainer: boolean }) => {
    const [interaction, setInteraction] = useState<GraphInteractionMetrics | null>(null);

    const handleInteraction = (metrics: GraphInteractionMetrics) => {
        setInteraction(metrics.isInteracting ? metrics : null);
    };

    const displayPrice = interaction ? `€${interaction.price.toFixed(2)}` : item.price;
    const displayChange = interaction ? `${interaction.change >= 0 ? '+' : ''}${interaction.change.toFixed(1)}%` : item.change;
    const displayIsGainer = interaction ? interaction.isPositive : isGainer;

    return (
        <Animated.View
            entering={FadeInDown.duration(400)}
            layout={Layout.springify()}
            className="bg-white/5 rounded-[6px] mb-3 overflow-hidden border border-white/10 shadow-sm backdrop-blur-md"
        >
            <View className="flex-row items-center" style={{ height: 120 }}>
                {/* Left Half: Image + Info (50%) */}
                <TouchableOpacity
                    onPress={() => router.push(`/card/${item.card_id}`)}
                    activeOpacity={0.8}
                    className="flex-1 flex-row p-3 items-center"
                    style={{ height: 120 }}
                >
                    <View
                        style={{ aspectRatio: 8 / 11 }}
                        className="h-24 bg-white/10 rounded-[4px] overflow-hidden border border-white/5 shadow-sm"
                    >
                        {getCardImage(item.image) ? (
                            <Image
                                source={{ uri: getCardImage(item.image)! }}
                                className="w-full h-full"
                                resizeMode="cover"
                            />
                        ) : (
                            <LinearGradient
                                colors={['rgba(255,255,255,0.1)', 'rgba(255,255,255,0.05)']}
                                className="w-full h-full items-center justify-center"
                            >
                                <FontAwesome name="image" size={20} color="rgba(255,255,255,0.2)" />
                            </LinearGradient>
                        )}
                    </View>

                    {/* Info */}
                    <View className="flex-1 ml-4 justify-center">
                        <Text className="text-white text-base font-bold" numberOfLines={1}>{item.name}</Text>
                        <View className="flex-row items-center mt-1 gap-x-2 flex-wrap">
                            <Text className="text-white/60 text-sm">{displayPrice}</Text>
                            <View className={`${displayIsGainer ? 'bg-green-500/20 border-green-500/30' : 'bg-red-500/20 border-red-500/30'} px-1.5 py-0.5 rounded border`}>
                                <Text className={`${displayIsGainer ? 'text-green-400' : 'text-red-400'} font-bold text-[10px] tracking-tight`}>
                                    {displayChange}
                                </Text>
                            </View>
                        </View>
                        {interaction && (
                            <Text className="text-gray-400 text-[10px] mt-1 font-bold uppercase tracking-wide">
                                {interaction.date}
                            </Text>
                        )}
                    </View>
                </TouchableOpacity>

                {/* Right Half: Interactive Price Trend (50%) - Seamless Background */}
                <View className="flex-1 overflow-hidden pr-4" style={{ height: 120 }}>
                    <InteractiveGraph
                        data={item.chartData}
                        width={(width - 40) / 2 - 24}
                        height={120} // Matches row height exactly
                        accentColor={isGainer ? '#4ade80' : '#f87171'}
                        onInteractionUpdate={handleInteraction}
                    />
                </View>
            </View>
        </Animated.View>
    );
};

export default function TrendsScreen() {
    const [selectedRange, setSelectedRange] = useState<TimeRange>('1W');
    const [trends, setTrends] = useState<{ gainers: MoverData[], losers: MoverData[] }>({ gainers: [], losers: [] });
    const [isLoading, setIsLoading] = useState(true);

    const ranges: TimeRange[] = ['1W', '1M', '3M', '1Y'];

    useEffect(() => {
        setIsLoading(true);
        fetch(`${API_URL}/api/trends?range=${selectedRange}`)
            .then(res => res.json())
            .then(async (data: { gainers: MoverData[], losers: MoverData[] }) => {
                // Enrich with TCGdex prices
                const allIds = [...data.gainers, ...data.losers].map(m => m.card_id);
                const tcgPrices = await getBatchPrices(allIds);

                const rangeMap = { '1W': 7, '1M': 30, '3M': 90, '1Y': 365 };
                const days = rangeMap[selectedRange];

                const enrichMovers = (moverList: MoverData[]) => {
                    return moverList.map(m => {
                        const pricing = tcgPrices[m.card_id]?.cardmarket;
                        if (!pricing) return m;

                        const currentPrice = pricing.avg || pricing.trend || 0;
                        const history = generateHistory(currentPrice, days, m.card_id, {
                            avg7: pricing.avg7,
                            avg30: pricing.avg30
                        });

                        // Calculate real change from history
                        const startVal = history[0].value;
                        const endVal = history[history.length - 1].value;
                        const absChange = endVal - startVal;
                        const pctChange = startVal !== 0 ? (absChange / startVal) * 100 : 0;

                        return {
                            ...m,
                            price: formatEuro(currentPrice),
                            change: `${pctChange >= 0 ? '+' : ''}${pctChange.toFixed(1)}%`,
                            changePct: pctChange,
                            chartData: history
                        };
                    });
                };

                setTrends({
                    gainers: enrichMovers(data.gainers).sort((a, b) => b.changePct - a.changePct),
                    losers: enrichMovers(data.losers).sort((a, b) => a.changePct - b.changePct)
                });
                setIsLoading(false);
            })
            .catch(err => {
                console.error(err);
                setIsLoading(false);
            });
    }, [selectedRange]);

    const currentMovers = trends;

    return (
        <View className="flex-1 bg-black">
            <LinearGradient
                colors={['rgba(213, 161, 0, 0.15)', 'rgba(0,0,0,0)']}
                style={StyleSheet.absoluteFill}
            />

            <SafeAreaView className="flex-1" edges={['top']}>
                <View className="px-6 pt-4 pb-2">
                    <View className="flex-row items-center justify-between mb-2">
                        <View>
                            <Text className="text-4xl font-bold text-white">Market</Text>
                            <Text className="text-yellow-400 text-3xl font-bold -mt-1">Trends</Text>
                        </View>
                        <TouchableOpacity className="bg-white/10 p-3 rounded-2xl border border-white/5">
                            <Ionicons name="notifications-outline" size={24} color="white" />
                        </TouchableOpacity>
                    </View>

                    <Text className="text-gray-500 text-[10px] font-bold tracking-[2px] uppercase mb-6">Real-time market volatility insights</Text>

                    {/* Time Range Selector */}
                    <View className="flex-row bg-white/5 p-1.5 rounded-[20px] mb-6 border border-white/5">
                        {ranges.map((range) => (
                            <TouchableOpacity
                                key={range}
                                onPress={() => setSelectedRange(range)}
                                className={`flex-1 py-2.5 rounded-[16px] items-center justify-center ${selectedRange === range ? 'bg-yellow-400' : ''}`}
                            >
                                <Text className={`font-bold text-xs ${selectedRange === range ? 'text-black' : 'text-gray-400'}`}>
                                    {range}
                                </Text>
                            </TouchableOpacity>
                        ))}
                    </View>
                </View>

                {isLoading && (
                    <View className="items-center py-4">
                        <ActivityIndicator color={Colors.dark.accentYellow} />
                        <Text className="text-gray-500 text-[10px] font-bold mt-2">Updating Market Data...</Text>
                    </View>
                )}

                <ScrollView
                    className="flex-1"
                    contentContainerStyle={{ paddingHorizontal: 20, paddingBottom: 160 }}
                    showsVerticalScrollIndicator={false}
                >
                    {/* Top Gainers Section */}
                    <Animated.View key={`gainers-${selectedRange}`} entering={FadeInRight.delay(100)} className="mb-8">
                        <View className="flex-row items-center mb-4">
                            <View className="bg-green-500/20 p-2 rounded-xl mr-3">
                                <Ionicons name="trending-up" size={18} color="#4ade80" />
                            </View>
                            <Text className="text-white text-lg font-bold">Top Gainers</Text>
                            <View className="flex-1 h-[1px] bg-white/10 ml-4" />
                        </View>

                        {currentMovers.gainers.map((item) => (
                            <TrendingCard key={item.card_id} item={item} isGainer={true} />
                        ))}
                    </Animated.View>

                    {/* Top Losers Section */}
                    <Animated.View key={`losers-${selectedRange}`} entering={FadeInRight.delay(200)}>
                        <View className="flex-row items-center mb-4">
                            <View className="bg-red-500/20 p-2 rounded-xl mr-3">
                                <Ionicons name="trending-down" size={18} color="#f87171" />
                            </View>
                            <Text className="text-white text-lg font-bold">Top Losers</Text>
                            <View className="flex-1 h-[1px] bg-white/10 ml-4" />
                        </View>

                        {currentMovers.losers.map((item) => (
                            <TrendingCard key={item.card_id} item={item} isGainer={false} />
                        ))}
                    </Animated.View>
                </ScrollView>
            </SafeAreaView>

            {/* Floating Action Button */}
        </View>
    );
}

const styles = StyleSheet.create({});
