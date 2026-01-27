import React, { useEffect } from 'react';
import { View, StyleSheet, useWindowDimensions } from 'react-native';
import Animated, {
    useSharedValue,
    useAnimatedStyle,
    withSpring,
    interpolate,
    Extrapolate,
    useDerivedValue,
} from 'react-native-reanimated';
import { Gesture, GestureDetector } from 'react-native-gesture-handler';
import { DeviceMotion } from 'expo-sensors';
import {
    Canvas,
    Group,
    Rect,
    LinearGradient,
    RadialGradient,
    vec,
    Blur,
    RoundedRect,
    Mask,
} from "@shopify/react-native-skia";

interface TiltCardProps {
    children: React.ReactNode;
}

const GESTURE_LIMIT = 45; // Extreme 3D tilt for touch
const SENSOR_LIMIT = 12;  // Subtle tilt for phone movement
const CARD_WIDTH = 280;
const CARD_HEIGHT = CARD_WIDTH / 0.727; // approximating 8/11
const RADIUS = 14;

export const TiltCard: React.FC<TiltCardProps> = ({ children }) => {
    const { width: screenWidth, height: screenHeight } = useWindowDimensions();

    // Shared values for rotation
    const rotateX = useSharedValue(0);
    const rotateY = useSharedValue(0);

    // Device motion listener
    useEffect(() => {
        let subscription: any;

        const startSensor = async () => {
            const isAvailable = await DeviceMotion.isAvailableAsync();
            if (!isAvailable) return;

            subscription = DeviceMotion.addListener((data) => {
                const { rotation } = data;
                if (rotation) {
                    // map phone tilt to the more subtle SENSOR_LIMIT
                    rotateX.value = withSpring(
                        interpolate(rotation.beta, [-Math.PI / 3, Math.PI / 3], [SENSOR_LIMIT, -SENSOR_LIMIT], Extrapolate.CLAMP),
                        { damping: 15, stiffness: 80 }
                    );
                    rotateY.value = withSpring(
                        interpolate(rotation.gamma, [-Math.PI / 3, Math.PI / 3], [-SENSOR_LIMIT, SENSOR_LIMIT], Extrapolate.CLAMP),
                        { damping: 15, stiffness: 80 }
                    );
                }
            });
            DeviceMotion.setUpdateInterval(16);
        };

        startSensor();

        return () => {
            subscription?.remove();
        };
    }, []);

    // Gesture handler
    const gesture = Gesture.Pan()
        .onUpdate((event) => {
            // map finger swipes to the dramatic GESTURE_LIMIT
            rotateY.value = withSpring(interpolate(event.translationX, [-screenWidth / 5, screenWidth / 5], [-GESTURE_LIMIT, GESTURE_LIMIT], Extrapolate.CLAMP));
            rotateX.value = withSpring(interpolate(event.translationY, [-screenHeight / 5, screenHeight / 5], [GESTURE_LIMIT, -GESTURE_LIMIT], Extrapolate.CLAMP));
        })
        .onEnd(() => {
            rotateX.value = withSpring(0);
            rotateY.value = withSpring(0);
        });

    const cardAnimatedStyle = useAnimatedStyle(() => {
        return {
            transform: [
                { perspective: 1200 },
                { rotateX: `${rotateX.value}deg` },
                { rotateY: `${rotateY.value}deg` },
            ],
        };
    });

    // Skia Derived Values for Gradients
    const glarePos = useDerivedValue(() => {
        // As rotateY increases (right tilt), glare moves left
        const x = interpolate(rotateY.value, [-GESTURE_LIMIT, GESTURE_LIMIT], [CARD_WIDTH * 1.5, -CARD_WIDTH * 0.5]);
        const y = interpolate(rotateX.value, [-GESTURE_LIMIT, GESTURE_LIMIT], [-CARD_HEIGHT * 0.5, CARD_HEIGHT * 1.5]);
        return vec(x, y);
    });

    const holoOpacity = useDerivedValue(() => {
        return interpolate(
            Math.abs(rotateX.value) + Math.abs(rotateY.value),
            [0, GESTURE_LIMIT * 2],
            [0, 0.35],
            Extrapolate.CLAMP
        );
    });

    const glareOpacity = useDerivedValue(() => {
        return interpolate(
            Math.abs(rotateX.value) + Math.abs(rotateY.value),
            [1, GESTURE_LIMIT * 2],
            [0, 0.4],
            Extrapolate.CLAMP
        );
    });

    const holoShift = useDerivedValue(() => {
        return interpolate(rotateY.value, [-GESTURE_LIMIT, GESTURE_LIMIT], [-100, 100]);
    });

    // Fix: Create derived values for Skia vectors instead of accessing .value in render
    const startPos = useDerivedValue(() => vec(holoShift.value, 0));
    const endPos = useDerivedValue(() => vec(holoShift.value + CARD_WIDTH, CARD_HEIGHT));

    return (
        <GestureDetector gesture={gesture}>
            <View style={styles.container}>
                <Animated.View style={[styles.card, cardAnimatedStyle]}>
                    <View style={styles.content}>
                        {children}
                        {/* Skia Overlays - Now inside clipped container */}
                        <View style={StyleSheet.absoluteFill} pointerEvents="none">
                            <Canvas style={StyleSheet.absoluteFill}>
                                <Group>
                                    {/* Holographic Rainbow Layer */}
                                    <Group blendMode="screen" opacity={holoOpacity}>
                                        <Rect x={-100} y={-100} width={CARD_WIDTH + 200} height={CARD_HEIGHT + 200}>
                                            <LinearGradient
                                                start={startPos}
                                                end={endPos}
                                                colors={[
                                                    "rgba(255, 0, 0, 0.4)",
                                                    "rgba(255, 255, 0, 0.4)",
                                                    "rgba(0, 255, 0, 0.4)",
                                                    "rgba(0, 255, 255, 0.4)",
                                                    "rgba(0, 0, 255, 0.4)",
                                                    "rgba(255, 0, 255, 0.4)",
                                                    "rgba(255, 0, 0, 0.4)",
                                                ]}
                                                positions={[0, 0.16, 0.33, 0.5, 0.66, 0.83, 1]}
                                            />
                                        </Rect>
                                    </Group>

                                    {/* Glare Layer */}
                                    <Group blendMode="overlay" opacity={glareOpacity}>
                                        <Rect x={0} y={0} width={CARD_WIDTH} height={CARD_HEIGHT}>
                                            <RadialGradient
                                                c={glarePos}
                                                r={CARD_WIDTH * 1.5}
                                                colors={["rgba(255, 255, 255, 0.6)", "transparent"]}
                                            />
                                        </Rect>
                                    </Group>
                                </Group>
                            </Canvas>
                        </View>
                    </View>
                </Animated.View>
            </View>
        </GestureDetector>
    );
};

const styles = StyleSheet.create({
    container: {
        alignItems: 'center',
        justifyContent: 'center',
        padding: 40,
    },
    card: {
        width: CARD_WIDTH,
        height: CARD_HEIGHT,
        backgroundColor: '#111',
        shadowColor: '#fff',
        shadowOffset: { width: 0, height: 10 },
        shadowOpacity: 0.2,
        shadowRadius: 30,
        elevation: 15,
        borderRadius: RADIUS,
    },
    content: {
        ...StyleSheet.absoluteFillObject,
        borderRadius: RADIUS,
        overflow: 'hidden',
    },
});
