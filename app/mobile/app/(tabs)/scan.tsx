
import { StyleSheet, TouchableOpacity, Alert, Platform, ActivityIndicator } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Text as ThemedText, View as ThemedView } from '@/components/Themed';
import { View, Text } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { useState, useRef, useEffect } from 'react';
import { useStore } from '../../store';
import FontAwesome from '@expo/vector-icons/FontAwesome';
import Ionicons from '@expo/vector-icons/Ionicons';
import { router } from 'expo-router';
import { Image, LayoutChangeEvent } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { API_URL } from '@/constants/Config';
import { StatusBar } from 'expo-status-bar';
import { getCardImage } from '@/utils/image';
import { getBatchPrices } from '@/utils/tcgdex';
import { formatEuro } from '@/utils/price';

export default function ScannerScreen() {
    const [permission, requestPermission] = useCameraPermissions();
    const [scanning, setScanning] = useState(false);
    const [scanResults, setScanResults] = useState<any>(null);
    const [selectedCardIdx, setSelectedCardIdx] = useState<number | null>(null);
    const [selectionIndex, setSelectionIndex] = useState<number>(0);
    const [statusMessage, setStatusMessage] = useState<string | null>(null);
    const [deletingCardIdx, setDeletingCardIdx] = useState<number | null>(null);
    const [containerLayout, setContainerLayout] = useState({ x: 0, width: 0 });
    const cameraRef = useRef<any>(null);
    const { addToPortfolio } = useStore();

    useEffect(() => {
        if (!permission?.granted) {
            requestPermission();
        }
    }, [permission]);

    if (!permission) {
        // Camera permissions are still loading
        return <View className="flex-1 bg-black" />;
    }

    if (!permission.granted) {
        // Camera permissions are not granted yet
        return (
            <View className="flex-1 bg-black items-center justify-center p-6">
                <Text className="text-white text-center mb-4">We need your permission to show the camera</Text>
                <TouchableOpacity onPress={requestPermission} className="bg-yellow-400 px-6 py-3 rounded-full">
                    <Text className="text-black font-bold">Grant Permission</Text>
                </TouchableOpacity>
            </View>
        );
    }

    const takePicture = async () => {
        if (!cameraRef.current || scanning) return;

        setScanning(true);
        try {
            const photo = await cameraRef.current.takePictureAsync({
                quality: 0.8,
                base64: false,
            });

            if (!photo) return;

            const formData = new FormData();
            // @ts-ignore
            formData.append('file', {
                uri: photo.uri,
                type: 'image/jpeg',
                name: 'scan.jpg',
            });

            const res = await fetch(`${API_URL}/api/identify?is_cropped=false`, {
                method: 'POST',
                body: formData,
            });

            if (!res.ok) {
                const errorText = await res.text();
                console.error(`API Error (${res.status}): ${errorText}`);
                throw new Error(`Connection Error: ${res.status}`);
            }
            const data = await res.json();

            if (data.match === false) {
                setStatusMessage("Card not detected. Try again.");
                setTimeout(() => setStatusMessage(null), 3000);
            } else {
                // Enrich with real TCGdex prices
                const allIds: string[] = [];
                data.cards.forEach((cardGroup: any) => {
                    cardGroup.top_matches.forEach((m: any) => allIds.push(m.card_id));
                });

                if (allIds.length > 0) {
                    const tcgPrices = await getBatchPrices(allIds);
                    data.cards.forEach((cardGroup: any) => {
                        cardGroup.top_matches.forEach((m: any) => {
                            const pricing = tcgPrices[m.card_id]?.cardmarket;
                            if (pricing) {
                                const current = pricing.avg || pricing.trend || 0;
                                if (current > 0) {
                                    m.price = formatEuro(current);
                                    const baseForChange = pricing.avg7 || pricing.avg30 || pricing.avg;
                                    const pctChange = (baseForChange && current) ? ((current - baseForChange) / baseForChange) * 100 : 0;
                                    m.change = `${pctChange >= 0 ? '+' : ''}${pctChange.toFixed(1)}%`;
                                    m.is_positive = pctChange >= 0;
                                }
                            } else {
                                console.log(`[Scanner] No TCGdex pricing for ${m.card_id}`);
                            }
                        });
                    });
                }

                setScanResults(data);
                if (data.cards.length === 1) {
                    setSelectedCardIdx(0);
                    setSelectionIndex(0);
                }
            }

        } catch (e: any) {
            console.error(e);
            setStatusMessage("Error connecting to server.");
            setTimeout(() => setStatusMessage(null), 3000);
        } finally {
            setScanning(false);
        }
    };

    const saveCard = () => {
        if (selectedCardIdx === null || !scanResults) return;
        const cardGroup = scanResults.cards[selectedCardIdx];
        const card = cardGroup.top_matches[selectionIndex];

        addToPortfolio({
            id: card.card_id,
            name: card.name || card.card_id,
            image: card.image,
            price: card.price,
            timestamp: Date.now()
        });

        dismissCard();
        setStatusMessage("Card added to portfolio!");
        setTimeout(() => setStatusMessage(null), 3000);
    };

    const saveAllCards = () => {
        if (!scanResults || !scanResults.cards.length) return;

        scanResults.cards.forEach((cardGroup: any) => {
            const card = cardGroup.top_matches[0];
            addToPortfolio({
                id: card.card_id,
                name: card.name || card.card_id,
                image: card.image,
                price: card.price,
                timestamp: Date.now()
            });
        });

        setScanResults(null);
        setSelectedCardIdx(null);
        setStatusMessage(`${scanResults.cards.length} cards added to portfolio!`);
        setTimeout(() => setStatusMessage(null), 3000);
    };

    const dismissCard = () => {
        if (selectedCardIdx === null || !scanResults) return;

        // Remove from current results
        const newCards = scanResults.cards.filter((_: any, idx: number) => idx !== selectedCardIdx);
        if (newCards.length > 0) {
            setScanResults({ ...scanResults, cards: newCards });
            setSelectedCardIdx(null);
        } else {
            setScanResults(null);
            setSelectedCardIdx(null);
        }
    };

    const removeSpecificResult = (idx: number) => {
        if (!scanResults) return;
        const newCards = scanResults.cards.filter((_: any, i: number) => i !== idx);
        if (newCards.length > 0) {
            setScanResults({ ...scanResults, cards: newCards });
        } else {
            setScanResults(null);
        }
        setDeletingCardIdx(null);
    };

    const currentCard = selectedCardIdx !== null ? scanResults.cards[selectedCardIdx].top_matches[selectionIndex] : null;
    const topMatches = selectedCardIdx !== null ? scanResults.cards[selectedCardIdx].top_matches : [];

    return (
        <View style={{ flex: 1, backgroundColor: 'black' }}>
            <StatusBar style="light" />
            <CameraView
                style={StyleSheet.absoluteFill}
                facing="back"
                ref={cameraRef}
            />

            {selectedCardIdx === null ? (
                <View style={{ flex: 1, backgroundColor: 'transparent' }}>
                    {/* Detection Markers - Card Images */}
                    {scanResults && scanResults.cards.map((cardGroup: any, idx: number) => (
                        <View
                            key={idx}
                            style={{
                                position: 'absolute',
                                left: `${cardGroup.center.x * 100}%`,
                                top: `${cardGroup.center.y * 100}%`,
                                transform: [{ translateX: -40 }, { translateY: -60 }],
                                zIndex: 20
                            }}
                        >
                            <TouchableOpacity
                                onPress={() => {
                                    if (deletingCardIdx !== null) {
                                        setDeletingCardIdx(null);
                                    } else {
                                        setSelectedCardIdx(idx);
                                        setSelectionIndex(0);
                                    }
                                }}
                                onLongPress={() => setDeletingCardIdx(idx)}
                                delayLongPress={500}
                                activeOpacity={0.8}
                            >
                                <View className="w-20 rounded overflow-hidden shadow-2xl bg-neutral-900" style={{ aspectRatio: 0.727 /* approximating 8/11 */ }}>
                                    {getCardImage(cardGroup.top_matches[0].image) && (
                                        <Image
                                            source={{ uri: getCardImage(cardGroup.top_matches[0].image)! }}
                                            className="w-full h-full"
                                            resizeMode="cover"
                                        />
                                    )}
                                </View>
                            </TouchableOpacity>

                            {deletingCardIdx === idx && (
                                <TouchableOpacity
                                    onPress={() => removeSpecificResult(idx)}
                                    className="absolute -top-3 -right-3 w-8 h-8 bg-red-600 rounded-full items-center justify-center border-2 border-white shadow-lg z-30"
                                >
                                    <Ionicons name="close" size={20} color="white" />
                                </TouchableOpacity>
                            )}
                        </View>
                    ))}

                    {/* Top Bar */}
                    <View
                        pointerEvents="box-none"
                        style={{ position: 'absolute', top: 60, left: 0, right: 0, paddingHorizontal: 24, flexDirection: 'row', justifyContent: 'flex-start', alignItems: 'center', zIndex: 10 }}
                    >
                        <TouchableOpacity
                            onPress={() => {
                                if (scanResults) setScanResults(null);
                                else router.back();
                            }}
                            className="w-10 h-10 bg-black/40 rounded-full items-center justify-center border border-white/10 backdrop-blur-md"
                        >
                            <Ionicons name="close" size={24} color="white" />
                        </TouchableOpacity>
                    </View>

                    {/* Status Message Overlay (Always available) */}
                    {statusMessage ? (
                        <View
                            pointerEvents="box-none"
                            style={{ position: 'absolute', top: 120, left: 0, right: 0, alignItems: 'center', zIndex: 100 }}
                        >
                            <View
                                className={`${statusMessage.includes('added') ? 'bg-green-500/90' : 'bg-red-500/90'} px-6 py-3 rounded-2xl border ${statusMessage.includes('added') ? 'border-green-400/30' : 'border-red-400/30'} backdrop-blur-xl shadow-2xl`}
                            >
                                <Text className="text-white font-bold text-sm">{statusMessage}</Text>
                            </View>
                        </View>
                    ) : null}


                    {/* Bottom Controls */}
                    <View
                        pointerEvents="box-none"
                        style={{ position: 'absolute', bottom: 50, left: 0, right: 0, alignItems: 'center' }}
                    >
                        <TouchableOpacity
                            onPress={scanResults ? saveAllCards : undefined}
                            activeOpacity={0.8}
                            disabled={!scanResults}
                            className={`${scanResults ? 'bg-green-500 shadow-green-500/50' : 'bg-yellow-400 shadow-yellow-400/50'} px-6 py-2 rounded-full mb-8 shadow-2xl`}
                        >
                            <Text style={{ color: 'black', fontWeight: '800', fontSize: 12, letterSpacing: 2 }}>
                                {scanResults ? `ACCEPT ALL (${scanResults.cards.length})` : "READY TO SCAN"}
                            </Text>
                        </TouchableOpacity>

                        <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'center', width: '100%' }}>
                            <TouchableOpacity
                                onPress={takePicture}
                                disabled={scanning}
                                style={{
                                    width: 84,
                                    height: 84,
                                    backgroundColor: 'white',
                                    borderRadius: 42,
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    borderWidth: 6,
                                    borderColor: 'rgba(255,255,255,0.2)',
                                }}
                            >
                                {scanning ? <ActivityIndicator color="black" /> : <View className="w-16 h-16 rounded-full bg-white border border-gray-200" />}
                            </TouchableOpacity>
                        </View>
                    </View>
                </View>
            ) : (
                <View className="flex-1 bg-black/80 p-6 justify-center">
                    <LinearGradient
                        colors={['rgba(213, 161, 0, 0.2)', 'rgba(0,0,0,0)']}
                        style={StyleSheet.absoluteFill}
                    />

                    <View className="bg-white/5 rounded-[20px] p-6 border border-white/10 shadow-3xl backdrop-blur-3xl">
                        <View className="items-center mb-6">
                            <TouchableOpacity
                                onPress={() => router.push(`/card/${currentCard?.card_id}`)}
                                activeOpacity={0.8}
                                className="w-40 bg-white/10 rounded-lg overflow-hidden border border-white/10 shadow-2xl mb-4"
                                style={{ aspectRatio: 0.727 /* approximating 8/11 */ }}
                            >
                                {getCardImage(currentCard.image) ? (
                                    <Image
                                        source={{ uri: getCardImage(currentCard.image)! }}
                                        className="w-full h-full"
                                        resizeMode="cover"
                                    />
                                ) : (
                                    <View className="w-full h-full items-center justify-center">
                                        <FontAwesome name="image" size={48} color="rgba(255,255,255,0.1)" />
                                    </View>
                                )}
                            </TouchableOpacity>
                            <Text className="text-yellow-400 font-extrabold text-[10px] tracking-[4px] uppercase mb-1">Card Identified</Text>
                            <Text className="text-white text-2xl font-bold text-center mb-1">{currentCard?.name}</Text>
                            <Text className="text-gray-500 font-bold text-xs tracking-widest uppercase">{currentCard?.card_id}</Text>
                        </View>

                        {/* Top Matches Selector with Scrubbing */}
                        <View className="mb-6">
                            <Text className="text-white/40 text-[10px] font-bold tracking-widest uppercase mb-3 text-center">Slide to select match:</Text>
                            <View
                                className="flex-row justify-center gap-2 py-2"
                                onLayout={(e) => {
                                    e.currentTarget.measure((x, y, width, height, pageX, pageY) => {
                                        setContainerLayout({ x: pageX, width });
                                    });
                                }}
                                onStartShouldSetResponder={() => true}
                                onResponderGrant={(evt) => {
                                    const itemWidth = 40;
                                    const gap = 8;
                                    const totalContentWidth = topMatches.length * itemWidth + (topMatches.length - 1) * gap;
                                    const startOffset = (containerLayout.width - totalContentWidth) / 2;
                                    const relativeX = evt.nativeEvent.pageX - containerLayout.x - startOffset;
                                    const index = Math.floor(relativeX / (itemWidth + gap));
                                    const constrainedIndex = Math.max(0, Math.min(topMatches.length - 1, index));
                                    if (constrainedIndex !== selectionIndex && index >= 0 && index < topMatches.length) {
                                        setSelectionIndex(constrainedIndex);
                                    }
                                }}
                                onResponderMove={(evt) => {
                                    const itemWidth = 40;
                                    const gap = 8;
                                    const totalContentWidth = topMatches.length * itemWidth + (topMatches.length - 1) * gap;
                                    const startOffset = (containerLayout.width - totalContentWidth) / 2;

                                    const relativeX = evt.nativeEvent.pageX - containerLayout.x - startOffset;
                                    const index = Math.floor(relativeX / (itemWidth + gap));

                                    const constrainedIndex = Math.max(0, Math.min(topMatches.length - 1, index));
                                    if (constrainedIndex !== selectionIndex && index >= 0 && index < topMatches.length) {
                                        setSelectionIndex(constrainedIndex);
                                    }
                                }}
                            >
                                {topMatches.map((m: any, idx: number) => (
                                    <View
                                        key={idx}
                                        className={`w-10 h-14 rounded-md overflow-hidden border-2 ${selectionIndex === idx ? 'border-yellow-400' : 'border-transparent'}`}
                                        style={{
                                            transform: [{ scale: selectionIndex === idx ? 1.2 : 1 }],
                                            opacity: selectionIndex === idx ? 1 : 0.4
                                        }}
                                    >
                                        {getCardImage(m.image) && (
                                            <Image
                                                source={{ uri: getCardImage(m.image)! }}
                                                className="w-full h-full"
                                                resizeMode="cover"
                                            />
                                        )}
                                    </View>
                                ))}
                            </View>
                        </View>

                        <View className="flex-row items-center justify-between bg-white/5 p-4 rounded-[20px] border border-white/5 mb-6">
                            <View>
                                <Text className="text-gray-500 text-[10px] font-bold tracking-widest uppercase mb-1">Market Price</Text>
                                <Text className="text-white text-2xl font-extrabold">{currentCard?.price}</Text>
                            </View>
                            {currentCard?.change && (
                                <View className={`${currentCard.is_positive ? 'bg-green-500/20 border-green-500/30' : 'bg-red-500/20 border-red-500/30'} px-3 py-1 rounded-full border`}>
                                    <Text className={`${currentCard.is_positive ? 'text-green-400' : 'text-red-400'} font-bold text-xs`}>{currentCard.change}</Text>
                                </View>
                            )}
                        </View>

                        <View className="flex-row gap-4">
                            <TouchableOpacity
                                onPress={dismissCard}
                                className="flex-1 py-4 bg-white/5 rounded-[20px] items-center border border-white/10 shadow-xl"
                            >
                                <Text className="text-white font-bold text-sm">Dismiss</Text>
                            </TouchableOpacity>
                            <TouchableOpacity
                                onPress={saveCard}
                                className="flex-2 py-4 bg-yellow-400 rounded-[20px] items-center shadow-2xl shadow-yellow-400/40"
                            >
                                <Text className="text-black font-extrabold text-sm px-6">Add to Portfolio</Text>
                            </TouchableOpacity>
                        </View>
                    </View>
                </View>
            )}
        </View>
    );
}
