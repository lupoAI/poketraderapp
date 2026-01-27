import React from 'react';
import { View, Text, Image, StyleSheet, ViewStyle, ImageStyle, TextStyle, TouchableOpacity } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import Ionicons from '@expo/vector-icons/Ionicons';

interface PremiumCardProps {
    id: string;
    name: string;
    image?: string;
    price: string;
    count?: number;
    isOwned?: boolean;
    showCheckmark?: boolean;
    containerStyle?: ViewStyle;
    onPress?: () => void;
    onLongPress?: () => void;
    isDeleting?: boolean;
    onDelete?: () => void;
}

export const PremiumCard: React.FC<PremiumCardProps> = ({
    id, name, image, price, count, isOwned, showCheckmark = true,
    containerStyle, onPress, onLongPress, isDeleting, onDelete
}) => {
    return (
        <TouchableOpacity
            style={[styles.container, containerStyle]}
            onPress={isDeleting ? onDelete : onPress}
            onLongPress={onLongPress}
            activeOpacity={0.8}
        >
            <View style={styles.imageContainer}>
                <Image
                    source={{ uri: image || 'https://via.placeholder.com/150' }}
                    style={styles.image}
                    resizeMode="cover"
                />
                <LinearGradient
                    colors={['transparent', 'rgba(0,0,0,0.8)']}
                    style={styles.gradient}
                />
                <View style={styles.infoBadge}>
                    <Text style={styles.priceText}>{price}</Text>
                </View>

                {isOwned && (showCheckmark || (!!count && count > 1)) && (
                    <View style={styles.countBadge}>
                        {!!count && count > 1 ? (
                            <Text style={styles.countText}>x{count}</Text>
                        ) : (
                            <Ionicons name="checkmark-sharp" size={12} color="black" />
                        )}
                    </View>
                )}

                {/* Fallback for when isOwned isn't passed but count is (older usage) */}
                {!isOwned && !!count && count > 1 && (
                    <View style={styles.countBadge}>
                        <Text style={styles.countText}>x{count}</Text>
                    </View>
                )}
            </View>

            {/* Delete Button Overlay */}
            {isDeleting && (
                <TouchableOpacity
                    style={styles.deleteOverlay}
                    onPress={onDelete}
                    activeOpacity={0.7}
                >
                    <Ionicons name="close" size={16} color="white" />
                </TouchableOpacity>
            )}
        </TouchableOpacity>
    );
};

const styles = StyleSheet.create({
    container: {
        width: '33.33%',
        padding: 4,
    },
    imageContainer: {
        aspectRatio: 0.727, // approximating 8/11
        width: '100%',
        backgroundColor: '#111',
        borderRadius: 6,
        overflow: 'hidden',
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.3,
        shadowRadius: 8,
        elevation: 5,
    },
    image: {
        ...StyleSheet.absoluteFillObject,
    },
    gradient: {
        position: 'absolute',
        bottom: 0,
        left: 0,
        right: 0,
        height: '40%',
    },
    infoBadge: {
        position: 'absolute',
        bottom: 12,
        left: 12,
        right: 12,
    },
    priceText: {
        color: '#FFCB05',
        fontSize: 12,
        fontWeight: 'bold',
        marginTop: 2,
    },
    countBadge: {
        position: 'absolute',
        bottom: 8,
        right: 8,
        backgroundColor: '#FFCB05',
        paddingHorizontal: 6,
        paddingVertical: 2,
        borderRadius: 4,
        borderWidth: 1,
        borderColor: 'rgba(0,0,0,0.2)',
    },
    countText: {
        color: 'black',
        fontSize: 10,
        fontWeight: '900',
    },
    deleteOverlay: {
        position: 'absolute',
        top: 0,
        right: 0,
        backgroundColor: '#ef4444',
        width: 24,
        height: 24,
        borderRadius: 12,
        alignItems: 'center',
        justifyContent: 'center',
        borderWidth: 2,
        borderColor: 'white',
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.3,
        shadowRadius: 4,
        elevation: 5,
        zIndex: 100,
    },
});
