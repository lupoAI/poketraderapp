import { useLocalSearchParams, useRouter, Stack } from 'expo-router';
import { View, Text, Image, TouchableOpacity, StyleSheet, ScrollView, Alert, ActivityIndicator, useWindowDimensions, TextInput } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import FontAwesome from '@expo/vector-icons/FontAwesome';
import Ionicons from '@expo/vector-icons/Ionicons';
import MaterialCommunityIcons from '@expo/vector-icons/MaterialCommunityIcons';
import { useStore } from '../../store';
import { API_URL } from '@/constants/Config';
import { LinearGradient } from 'expo-linear-gradient';
import { TiltCard } from '@/components/TiltCard';
import { InteractiveGraph, GraphInteractionMetrics } from '@/components/InteractiveGraph';
import { getCardImage, getLogoImage } from '@/utils/image';
import { generateHistory } from '@/utils/price';
import { useState, useEffect, useMemo } from 'react';
import Colors, { TYPE_COLORS } from '../../constants/Colors';
import Animated, {
    useSharedValue,
    useDerivedValue,
    useAnimatedStyle,
    withSpring,
    withTiming,
    interpolate,
    Extrapolate,
    runOnJS,
    useAnimatedReaction
} from 'react-native-reanimated';
import { GestureDetector, Gesture } from 'react-native-gesture-handler';
import * as Haptics from 'expo-haptics';

const AnimatedTextInput = Animated.createAnimatedComponent(TextInput);
const AnimatedText = Animated.createAnimatedComponent(Text);

export default function CardDetailScreen() {
    const { id } = useLocalSearchParams();
    const router = useRouter();
    const { width: screenWidth } = useWindowDimensions();
    const { portfolio, removeOneFromPortfolio, removeFromPortfolio, addToPortfolio } = useStore();

    const [cardData, setCardData] = useState<any>(null);
    const [historicalPrices, setHistoricalPrices] = useState<any[]>([]);
    const [selectedRange, setSelectedRange] = useState<'1W' | '1M' | '3M' | '1Y'>('1M');
    const [loading, setLoading] = useState(true);

    // State for interactive UI updates
    const [displayPrice, setDisplayPrice] = useState<number>(0);
    const [displayChange, setDisplayChange] = useState<number>(0);
    const [displayAbsChange, setDisplayAbsChange] = useState<number>(0);
    const [displayDate, setDisplayDate] = useState<string>('');
    const [displayIsInteracting, setDisplayIsInteracting] = useState(false);
    const [displayIsPositive, setDisplayIsPositive] = useState(true);

    const chartWidth = screenWidth - 80;

    const handleGraphInteraction = (metrics: GraphInteractionMetrics) => {
        setDisplayPrice(metrics.price);
        setDisplayChange(metrics.change);
        setDisplayAbsChange(metrics.absChange);
        setDisplayIsPositive(metrics.isPositive);
        setDisplayDate(metrics.date);
        setDisplayIsInteracting(metrics.isInteracting);
    };

    // Price calculations for non-interacting state
    const currentPrice = historicalPrices.length > 0 ? historicalPrices[historicalPrices.length - 1].value : 0;
    const initialPriceValue = historicalPrices.length > 0 ? historicalPrices[0].value : 0;
    const priceChange = currentPrice - initialPriceValue;
    const priceChangePct = initialPriceValue !== 0 ? (priceChange / initialPriceValue) * 100 : 0;
    const isPricePositive = priceChange >= 0;

    // Find all copies of this card in portfolio
    const copies = portfolio.filter(c => c.id === id);
    const portfolioInfo = copies[0];

    useEffect(() => {
        const fetchCardDetails = async () => {
            // If we already have full_data in the portfolio, use it
            if (portfolioInfo?.full_data) {
                setCardData(portfolioInfo.full_data);
                setLoading(false);
                return;
            }

            try {
                const response = await fetch(`https://api.tcgdex.net/v2/en/cards/${id}`);
                if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                const data = await response.json();
                setCardData(data);
            } catch (error) {
                console.error("Error fetching card details from TCGdex API:", error);
            } finally {
                setLoading(false);
            }
        };

        if (id) {
            fetchCardDetails();
        }
    }, [id, portfolioInfo?.full_data]);

    useEffect(() => {
        const fetchHistoricalPrices = async () => {
            if (!cardData?.pricing?.cardmarket) {
                // Generic fallback if no real pricing
                const rangeMap = { '1W': 7, '1M': 30, '3M': 90, '1Y': 365 };
                setHistoricalPrices(generateHistory(10.0, rangeMap[selectedRange], id as string));
                return;
            }

            const pricing = cardData.pricing.cardmarket;
            const currentPrice = pricing.avg || pricing.trend || 0;
            const rangeMap = { '1W': 7, '1M': 30, '3M': 90, '1Y': 365 };
            const days = rangeMap[selectedRange];

            const history = generateHistory(
                currentPrice,
                days,
                id as string,
                {
                    avg7: pricing.avg7,
                    avg30: pricing.avg30
                }
            );
            setHistoricalPrices(history);
        };

        if (id && cardData) {
            fetchHistoricalPrices();
        }
    }, [id, selectedRange, cardData]);

    if (!portfolioInfo && !loading && !cardData) {
        return (
            <View className="flex-1 bg-black items-center justify-center">
                <Stack.Screen options={{ headerShown: false }} />
                <Text className="text-white">Card not found</Text>
                <TouchableOpacity onPress={() => router.back()} className="mt-4">
                    <Text className="text-yellow-400">Back to Portfolio</Text>
                </TouchableOpacity>
            </View>
        );
    }

    const handleDeleteOne = () => {
        if (copies.length > 1) {
            removeOneFromPortfolio(id as string);
        } else {
            Alert.alert(
                "Delete Card",
                "Are you sure you want to remove the last copy of this card?",
                [
                    { text: "Cancel", style: "cancel" },
                    {
                        text: "Delete",
                        style: "destructive",
                        onPress: () => {
                            removeOneFromPortfolio(id as string);
                            router.back();
                        }
                    }
                ]
            );
        }
    };

    const handleAddOne = () => {
        if (portfolioInfo) {
            addToPortfolio({
                ...portfolioInfo,
                timestamp: Date.now()
            });
        }
    };

    const handleDeleteAll = () => {
        Alert.alert(
            "Delete All",
            `Are you sure you want to remove all ${copies.length} copies of this card?`,
            [
                { text: "Cancel", style: "cancel" },
                {
                    text: "Delete All",
                    style: "destructive",
                    onPress: () => {
                        removeFromPortfolio(id as string);
                        router.back();
                    }
                }
            ]
        );
    };

    // Use high-quality image if available, otherwise fallback to local/portfolio image
    const { placeholder } = useLocalSearchParams() as { placeholder?: string };

    const lowResImage = getCardImage(portfolioInfo?.image || placeholder, 'low');

    // TCGdex API returns base image URL in 'image' field
    const highResImage = getCardImage(cardData?.image, 'high');
    const [highResLoaded, setHighResLoaded] = useState(false);

    const [aspectRatio, setAspectRatio] = useState(0.727); // default 8/11

    useEffect(() => {
        if (lowResImage) {
            Image.getSize(lowResImage, (width, height) => {
                if (width && height) {
                    setAspectRatio(width / height);
                }
            }, (error) => {
                console.warn("Failed to get low-res image size:", error);
            });
        }
    }, [lowResImage]);

    useEffect(() => {
        if (highResImage) {
            Image.getSize(highResImage, (width, height) => {
                if (width && height) {
                    setAspectRatio(width / height);
                }
            }, (error) => {
                console.warn("Failed to get high-res image size:", error);
            });
        }
    }, [highResImage]);

    // Dynamic color based on card type
    const cardType = cardData?.types?.[0] || 'Colorless';
    const accentColor = TYPE_COLORS[cardType] || '#FFCB05';

    return (
        <View className="flex-1 bg-black">
            <Stack.Screen options={{ headerShown: false }} />
            <LinearGradient
                colors={[`${accentColor}33`, 'rgba(0,0,0,0)']} // 33 is ~20% opacity in hex
                style={StyleSheet.absoluteFill}
            />

            <SafeAreaView className="flex-1" edges={['top', 'left', 'right']}>
                {/* Header */}
                <View className="flex-row justify-between items-center px-6 pt-4 pb-2">
                    <TouchableOpacity
                        onPress={() => router.back()}
                        className="w-10 h-10 bg-white/10 rounded-full items-center justify-center border border-white/5"
                        style={{ paddingRight: 2 }} // Optical nudge for back arrow
                    >
                        <Ionicons name="chevron-back" size={24} color="white" />
                    </TouchableOpacity>
                    <Text className="text-white font-bold text-2xl" numberOfLines={1}>{cardData?.name || portfolioInfo?.name || "Card Details"}</Text>
                    <View className="w-10" />
                </View>

                {loading ? (
                    <View className="flex-1 items-center justify-center">
                        <ActivityIndicator color="#D5A100" size="large" />
                    </View>
                ) : (
                    <ScrollView
                        className="flex-1"
                        contentContainerStyle={{ flexGrow: 1, paddingBottom: 60 }}
                        showsVerticalScrollIndicator={false}
                    >
                        <View className="items-center px-6 pt-6">
                            {/* 3D Tilting Card */}
                            <View className="mb-6 shadow-2xl shadow-yellow-500/10">
                                <TiltCard>
                                    <View
                                        style={{ aspectRatio, width: 280, borderRadius: 14 }}
                                        className="overflow-hidden border border-white/20 bg-neutral-900"
                                    >
                                        {/* Low-res placeholder / fallback */}
                                        {lowResImage && (
                                            <Image
                                                source={{ uri: lowResImage }}
                                                className="w-full h-full absolute"
                                                resizeMode="contain"
                                                blurRadius={highResImage ? 0.5 : 0}
                                            />
                                        )}

                                        {/* High-res image overlay */}
                                        {highResImage && (
                                            <Image
                                                source={{ uri: highResImage }}
                                                className={`w-full h-full ${highResLoaded ? 'opacity-100' : 'opacity-0'}`}
                                                resizeMode="contain"
                                                onLoad={() => setHighResLoaded(true)}
                                            />
                                        )}
                                    </View>
                                </TiltCard>
                            </View>

                            {/* Price and Inventory Block */}
                            <View className="w-full bg-white/5 p-6 rounded-[32px] border border-white/5 mb-6">
                                <View className="mb-8">
                                    <View className="flex-row items-baseline mb-1">
                                        <Text className="text-white text-4xl font-extrabold">€</Text>
                                        <Text className="text-white text-4xl font-extrabold ml-1">
                                            {Math.floor(displayIsInteracting ? displayPrice : currentPrice)}
                                        </Text>
                                        <Text className="text-white text-xl font-bold opacity-60">
                                            .{((displayIsInteracting ? displayPrice : currentPrice) % 1).toFixed(2).split('.')[1]}
                                        </Text>
                                    </View>

                                    <View className="flex-row items-center flex-wrap">
                                        <Text className={`${(displayIsInteracting ? (displayAbsChange >= 0) : isPricePositive) ? 'text-green-400' : 'text-red-400'} font-bold text-sm mr-2`}>
                                            {(displayIsInteracting ? (displayAbsChange >= 0) : isPricePositive) ? '+' : '-'}€{Math.abs(displayIsInteracting ? displayAbsChange : priceChange).toFixed(2)}
                                        </Text>
                                        <View className={`${(displayIsInteracting ? (displayChange >= 0) : isPricePositive) ? 'bg-green-500/20 border-green-500/30' : 'bg-red-500/20 border-red-500/30'} px-2 py-0.5 rounded border flex-row items-center mr-2`}>
                                            <FontAwesome
                                                name={(displayIsInteracting ? (displayChange >= 0) : isPricePositive) ? "caret-up" : "caret-down"}
                                                size={12}
                                                color={(displayIsInteracting ? (displayChange >= 0) : isPricePositive) ? '#4ADE80' : '#F87171'}
                                            />
                                            <Text className={`${(displayIsInteracting ? (displayChange >= 0) : isPricePositive) ? 'text-green-400' : 'text-red-400'} font-bold text-xs ml-1`}>
                                                {(displayIsInteracting ? Math.abs(displayChange) : Math.abs(priceChangePct)).toFixed(2)}%
                                            </Text>
                                        </View>
                                        <Text className="text-gray-500 text-xs">
                                            · {displayIsInteracting ? displayDate : `Past ${selectedRange === '1W' ? 'Week' : selectedRange === '1M' ? 'Month' : selectedRange === '3M' ? '3 Months' : 'Year'}`}
                                        </Text>
                                    </View>
                                </View>

                                <View className="mb-4 h-40 w-full items-center justify-center">
                                    <InteractiveGraph
                                        data={historicalPrices}
                                        width={chartWidth}
                                        height={140}
                                        accentColor={accentColor}
                                        onInteractionUpdate={handleGraphInteraction}
                                    />
                                </View>

                                {/* Range Selectors */}
                                <View className="flex-row gap-1 mb-6 bg-white/5 p-1 rounded-2xl w-full border border-white/5">
                                    {(['1W', '1M', '3M', '1Y'] as const).map((r) => (
                                        <TouchableOpacity
                                            key={r}
                                            onPress={() => setSelectedRange(r)}
                                            className={`${selectedRange === r ? 'bg-white/10' : ''} flex-1 py-2 rounded-xl items-center`}
                                        >
                                            <Text className={`${selectedRange === r ? 'text-white' : 'text-gray-500'} font-bold text-[10px]`}>{r}</Text>
                                        </TouchableOpacity>
                                    ))}
                                </View>

                                <View className="h-[1px] bg-white/5 w-full mb-6" />

                                <View className="flex-row justify-between items-center">
                                    <View>
                                        <Text className="text-gray-500 text-[10px] font-bold tracking-widest uppercase">Your Collection</Text>
                                        <Text className="text-white font-bold mt-1">{copies.length} Copies Owned</Text>
                                    </View>
                                    <View className="flex-row gap-4">
                                        <TouchableOpacity onPress={handleDeleteOne} className="w-10 h-10 items-center justify-center bg-red-500/10 rounded-full border border-red-500/10">
                                            <MaterialCommunityIcons name="minus" size={20} color="#EF4444" />
                                        </TouchableOpacity>
                                        <TouchableOpacity
                                            onPress={handleAddOne}
                                            className="w-10 h-10 items-center justify-center bg-yellow-400/10 rounded-full border border-yellow-400/10"
                                        >
                                            <MaterialCommunityIcons name="plus" size={20} color="#D5A100" />
                                        </TouchableOpacity>
                                    </View>
                                </View>
                            </View>

                            {/* Detailed Info Grid */}
                            {(cardData || portfolioInfo) && (
                                <View className="w-full gap-4 mb-8">
                                    {/* Primary Attributes (New Section) */}
                                    <View className="flex-row gap-4">
                                        <View className="flex-1 bg-white/5 p-4 rounded-2xl border border-white/5">
                                            <Text className="text-gray-500 text-[10px] font-bold tracking-widest uppercase mb-1">Type</Text>
                                            <View className="flex-row gap-2 mt-1">
                                                {cardData?.types?.map((t: string) => (
                                                    <View key={t} className="bg-white/10 px-2 py-1 rounded-lg border border-white/10">
                                                        <Text className="text-white text-xs font-bold">{t}</Text>
                                                    </View>
                                                )) || <Text className="text-white font-bold">---</Text>}
                                            </View>
                                        </View>
                                        <View className="w-24 bg-white/5 p-4 rounded-2xl border border-white/5 items-center justify-center">
                                            <Text className="text-gray-500 text-[10px] font-bold tracking-widest uppercase mb-1">HP</Text>
                                            <Text className="text-white font-bold text-base">{cardData?.hp || "---"}</Text>
                                        </View>
                                    </View>

                                    {/* Set & Rarity */}
                                    <View className="flex-row gap-4">
                                        <TouchableOpacity
                                            onPress={() => cardData?.set?.id && router.push(`/set/${cardData.set.id}` as any)}
                                            className="flex-1 bg-white/5 p-4 rounded-2xl border border-white/5 flex-row justify-between items-center"
                                        >
                                            <View className="flex-1 mr-2">
                                                <Text className="text-gray-500 text-[10px] font-bold tracking-widest uppercase mb-1">Set</Text>
                                                <Text className="text-white font-bold" numberOfLines={1}>{cardData?.set?.name || "---"}</Text>
                                            </View>
                                            {cardData?.set?.logo && (
                                                <Image
                                                    source={{ uri: getLogoImage(cardData.set.logo) }}
                                                    style={{ width: 40, height: 20 }}
                                                    resizeMode="contain"
                                                />
                                            )}
                                        </TouchableOpacity>
                                        <View className="flex-1 bg-white/5 p-4 rounded-2xl border border-white/5 items-center justify-center">
                                            <Text className="text-gray-500 text-[10px] font-bold tracking-widest uppercase mb-1">Rarity</Text>
                                            <Text className="text-white font-bold text-center" numberOfLines={1}>{cardData?.rarity || "---"}</Text>
                                        </View>
                                    </View>

                                    {/* Types & Weaknesses/Resistances */}
                                    <View className="flex-row gap-4">
                                        <View className="flex-1 bg-white/5 p-4 rounded-2xl border border-white/5">
                                            <Text className="text-gray-500 text-[10px] font-bold tracking-widest uppercase mb-2">Types</Text>
                                            <View className="flex-row gap-2">
                                                {cardData.types?.map((t: string) => (
                                                    <View key={t} className="bg-white/10 px-2 py-1 rounded-lg">
                                                        <Text className="text-white text-xs font-bold">{t}</Text>
                                                    </View>
                                                ))}
                                            </View>
                                        </View>

                                        {(cardData?.weaknesses?.length > 0 || cardData?.resistances?.length > 0) && (
                                            <View className="flex-1 bg-white/5 p-4 rounded-2xl border border-white/5">
                                                {cardData?.weaknesses?.length > 0 && (
                                                    <View className="mb-2">
                                                        <Text className="text-gray-500 text-[10px] font-bold tracking-widest uppercase mb-1">Weakness</Text>
                                                        <View className="flex-row gap-2">
                                                            {cardData.weaknesses.map((w: any, idx: number) => (
                                                                <Text key={idx} className="text-white text-xs font-bold">{w.type} {w.value}</Text>
                                                            ))}
                                                        </View>
                                                    </View>
                                                )}
                                                {cardData?.resistances?.length > 0 && (
                                                    <View>
                                                        <Text className="text-gray-500 text-[10px] font-bold tracking-widest uppercase mb-1">Resistance</Text>
                                                        <View className="flex-row gap-2">
                                                            {cardData.resistances.map((r: any, idx: number) => (
                                                                <Text key={idx} className="text-white text-xs font-bold">{r.type} {r.value}</Text>
                                                            ))}
                                                        </View>
                                                    </View>
                                                )}
                                            </View>
                                        )}
                                    </View>

                                    {/* Attacks */}
                                    {cardData?.attacks && cardData.attacks.length > 0 && (
                                        <View className="bg-white/5 p-5 rounded-[24px] border border-white/5">
                                            <Text className="text-gray-500 text-[10px] font-bold tracking-widest uppercase mb-4">Moves</Text>
                                            {cardData.attacks.map((attack: any, idx: number) => (
                                                <View key={idx} className={idx > 0 ? "mt-4 pt-4 border-t border-white/5" : ""}>
                                                    <View className="flex-row justify-between items-center mb-1">
                                                        <Text className="text-white font-bold text-base">{attack.name}</Text>
                                                        {attack.damage && (
                                                            <Text className="text-yellow-400 font-bold text-base">{attack.damage}</Text>
                                                        )}
                                                    </View>
                                                    <Text className="text-white/60 text-xs leading-5">{attack.effect}</Text>
                                                </View>
                                            ))}
                                        </View>
                                    )}

                                    {/* Description */}
                                    {cardData?.description && (
                                        <View className="bg-white/5 p-5 rounded-[24px] border border-white/5">
                                            <Text className="text-gray-500 text-[10px] font-bold tracking-widest uppercase mb-2">Background</Text>
                                            <Text className="text-white/40 italic text-xs leading-5">{cardData.description}</Text>
                                        </View>
                                    )}

                                    {/* Illustrator */}
                                    {cardData?.illustrator && (
                                        <View className="bg-white/5 p-4 rounded-2xl border border-white/5 flex-row justify-between items-center">
                                            <Text className="text-gray-500 text-[10px] font-bold tracking-widest uppercase">Illustrator</Text>
                                            <Text className="text-white text-xs font-bold">{cardData.illustrator}</Text>
                                        </View>
                                    )}
                                </View>
                            )}

                            {/* Actions */}
                            <TouchableOpacity
                                onPress={handleDeleteAll}
                                className="w-full bg-red-500/10 py-5 rounded-[24px] items-center border border-red-500/20"
                            >
                                <Text className="text-red-500 font-bold">Remove from Portfolio</Text>
                            </TouchableOpacity>
                        </View>
                    </ScrollView>
                )}
            </SafeAreaView>
        </View >
    );
}
