import React from 'react';
import { View, Text, Image, TouchableOpacity, StyleSheet } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { useStore } from '../store';
import { getCardImage, getLogoImage } from '../utils/image';

interface SetCardProps {
    id: string;
    name: string;
    logo: string;
    topCards?: string[];
    topCardImages?: string[];
    ownedCount: number;
    totalCount: number;
    onPress: () => void;
    onCardPress?: (cardId: string, image: string) => void;
}

export const SetCard: React.FC<SetCardProps> = ({ id, name, logo, topCards = [], topCardImages = [], ownedCount, totalCount, onPress, onCardPress }) => {
    const { portfolio, setTopCards } = useStore();
    const percentage = totalCount > 0 ? (ownedCount / totalCount) * 100 : 0;

    // We show up to 3 cards in the podium
    // We show up to 3 cards in the podium
    const podiumCards = topCards.slice(0, 3);
    const podiumImages = topCardImages?.slice(0, 3) || [];

    // Visually reorder: [Second, First, Third] so that First is in the Center (index 1)
    const displayCards = [...podiumCards];
    const displayImages = [...podiumImages];

    if (displayCards.length >= 2) {
        // Swap rank #1 and rank #2
        [displayCards[0], displayCards[1]] = [displayCards[1], displayCards[0]];
        [displayImages[0], displayImages[1]] = [displayImages[1], displayImages[0]];
    }

    const isPodiumVisible = podiumCards.length > 0 && podiumImages.length > 0;

    return (
        <TouchableOpacity
            onPress={onPress}
            className="flex-1 m-2 bg-white/5 rounded-[32px] border border-white/5 overflow-hidden"
            activeOpacity={0.8}
        >
            <View className={`p-4 items-center ${!isPodiumVisible ? 'justify-center min-h-[180px]' : ''}`}>
                {/* Logo Section */}
                <View className={`${isPodiumVisible ? 'h-16 mb-2' : 'h-28 mb-4'} w-full items-center justify-center`}>
                    {getLogoImage(logo) && (
                        <Image
                            source={{ uri: getLogoImage(logo)! }}
                            className="w-full h-full"
                            resizeMode="contain"
                        />
                    )}
                </View>

                {/* Podium Section */}
                {isPodiumVisible && (
                    <View className="h-28 w-full mb-4 items-center justify-center">
                        <View className="flex-row items-center justify-center w-full">
                            {displayCards.map((cardId, index) => {
                                const imagePath = displayImages[index];
                                const resolvedUrl = getCardImage(imagePath);
                                const isCached = true; // getCardImage handles store lookup

                                // Staggered layout - Balanced spread and rotation
                                const offset = (index - 1) * 35;
                                const rotation = (index - 1) * 8;
                                const zIndex = index === 1 ? 2 : 1;
                                const scale = index === 1 ? 1 : 0.88;

                                return (
                                    <TouchableOpacity
                                        key={cardId}
                                        onPress={() => {
                                            if (imagePath) onCardPress?.(cardId, getCardImage(imagePath)!);
                                        }}
                                        activeOpacity={0.9}
                                        style={{
                                            position: 'absolute',
                                            left: '50%',
                                            marginLeft: offset - 30, // -30 to center (approx half width)
                                            transform: [{ rotate: `${rotation}deg` }, { scale }],
                                            zIndex
                                        }}
                                        className="shadow-2xl"
                                    >
                                        {resolvedUrl && (
                                            <Image
                                                source={{ uri: resolvedUrl }}
                                                style={{ width: 60, height: 84, borderRadius: 4 }}
                                                resizeMode="cover"
                                            />
                                        )}
                                    </TouchableOpacity>
                                );
                            })}
                        </View>
                    </View>
                )}

                <Text className="text-white font-bold text-center text-[10px] mb-2 px-1" numberOfLines={1}>
                    {name}
                </Text>

                <View className="w-full px-1">
                    <View className="flex-row justify-between items-center mb-1">
                        <Text className="text-gray-500 text-[8px] font-bold uppercase tracking-wider">
                            Progress
                        </Text>
                        <Text className="text-yellow-400 text-[8px] font-bold">
                            {ownedCount}/{totalCount}
                        </Text>
                    </View>
                    <View className="h-1 bg-white/10 rounded-full overflow-hidden">
                        <LinearGradient
                            colors={['#FFCB05', '#D5A100']}
                            start={{ x: 0, y: 0 }}
                            end={{ x: 1, y: 0 }}
                            style={{ width: `${percentage}%`, height: '100%' }}
                        />
                    </View>
                </View>
            </View>
        </TouchableOpacity>
    );
};
