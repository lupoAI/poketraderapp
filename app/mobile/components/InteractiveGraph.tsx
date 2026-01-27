import React, { useMemo } from 'react';
import { View, StyleSheet, ActivityIndicator, Text } from 'react-native';
import { LineChart } from 'react-native-wagmi-charts';
import { GestureDetector, Gesture } from 'react-native-gesture-handler';
import Animated, {
    useSharedValue,
    useDerivedValue,
    useAnimatedStyle,
    withTiming,
    runOnJS,
    useAnimatedReaction
} from 'react-native-reanimated';
import * as Haptics from 'expo-haptics';

export interface GraphInteractionMetrics {
    price: number;
    change: number;
    absChange: number;
    isPositive: boolean;
    date: string;
    isInteracting: boolean;
    fingers: number;
    leftIndex: number;
    rightIndex: number;
}

interface InteractiveGraphProps {
    data: { timestamp: number, value: number }[];
    width: number;
    height: number;
    accentColor: string;
    onInteractionUpdate?: (metrics: GraphInteractionMetrics) => void;
}

export const InteractiveGraph = ({
    data,
    width,
    height,
    accentColor,
    onInteractionUpdate
}: InteractiveGraphProps) => {
    // Interactive Chart State
    const activeFingers = useSharedValue(0);
    const fingerX1 = useSharedValue(0);
    const fingerX2 = useSharedValue(0);
    const isInteracting = useSharedValue(false);

    // Local state for render-phase logic (fixes Reanimated warning)
    const [graphState, setGraphState] = React.useState({
        isInteracting: false,
        isPositive: true
    });

    const leftIndex = useDerivedValue(() => {
        if (!data.length) return 0;
        const x = Math.min(fingerX1.value, fingerX2.value);
        const idx = Math.floor((x / width) * data.length);
        return Math.max(0, Math.min(idx, data.length - 1));
    });

    const rightIndex = useDerivedValue(() => {
        if (!data.length) return 0;
        const x = Math.max(fingerX1.value, fingerX2.value);
        const idx = Math.floor((x / width) * data.length);
        return Math.max(0, Math.min(idx, data.length - 1));
    });

    const interactiveMetrics = useDerivedValue(() => {
        if (!data.length) return { price: 0, change: 0, absChange: 0, isPositive: true };

        let startVal = data[0].value;
        let endVal = data[data.length - 1].value;

        if (activeFingers.value === 1) {
            endVal = data[rightIndex.value].value;
        } else if (activeFingers.value === 2) {
            startVal = data[leftIndex.value].value;
            endVal = data[rightIndex.value].value;
        }

        const absChange = endVal - startVal;
        const changePct = startVal !== 0 ? (absChange / startVal) * 100 : 0;
        const isPositive = absChange >= 0;

        return {
            price: endVal,
            change: changePct,
            absChange: absChange || 0,
            isPositive
        };
    });

    const notifyUpdate = (metrics: any) => {
        if (onInteractionUpdate) {
            onInteractionUpdate(metrics);
        }
    };

    useAnimatedReaction(
        () => ({
            interacting: isInteracting.value,
            metrics: interactiveMetrics.value,
            left: leftIndex.value,
            right: rightIndex.value,
            fingers: activeFingers.value
        }),
        (res, previous) => {
            // 1. Sync visual state to JS if changed (Safe Render)
            if (res.interacting !== previous?.interacting || res.metrics.isPositive !== previous?.metrics.isPositive) {
                runOnJS(setGraphState)({
                    isInteracting: res.interacting,
                    isPositive: res.metrics.isPositive
                });
            }

            // 2. Notify Parent & Haptics
            if (res.interacting || (previous && previous.interacting)) {
                // Formatting date logic
                let displayDate = '';
                if (data.length > 0) {
                    const formatDate = (ts: number) => {
                        return new Date(ts).toLocaleDateString('en-US', {
                            month: 'short',
                            day: 'numeric',
                            year: 'numeric'
                        });
                    };

                    if (res.fingers === 2) {
                        const start = data[res.left].timestamp;
                        const end = data[res.right].timestamp;
                        displayDate = `${formatDate(start)} - ${formatDate(end)}`;
                    } else if (res.fingers === 1) {
                        const date = data[res.right].timestamp;
                        displayDate = formatDate(date);
                    }
                }

                runOnJS(notifyUpdate)({
                    ...res.metrics,
                    date: displayDate,
                    isInteracting: res.interacting,
                    fingers: res.fingers,
                    leftIndex: res.left,
                    rightIndex: res.right
                });

                // Trigger haptic feedback when index changes
                if (res.interacting && (res.left !== previous?.left || res.right !== previous?.right)) {
                    runOnJS(Haptics.selectionAsync)();
                }
            }
        },
        [data, width]
    );

    const gesture = useMemo(() => Gesture.Pan()
        .minDistance(0)
        .onTouchesDown((e) => {
            isInteracting.value = true;
            activeFingers.value = e.numberOfTouches;
            const x = Math.max(0, Math.min(e.allTouches[0].x, width));
            fingerX1.value = x;
            fingerX2.value = x;
        })
        .onUpdate((e) => {
            if (activeFingers.value === 1) {
                const x = Math.max(0, Math.min(e.x, width));
                fingerX1.value = x;
                fingerX2.value = x;
            }
        })
        .onTouchesMove((e) => {
            activeFingers.value = e.numberOfTouches;
            if (e.numberOfTouches >= 2) {
                const x1 = Math.max(0, Math.min(e.allTouches[0].x, width));
                const x2 = Math.max(0, Math.min(e.allTouches[1].x, width));
                fingerX1.value = Math.min(x1, x2);
                fingerX2.value = Math.max(x1, x2);
            }
        })
        .onTouchesUp((e) => {
            if (e.numberOfTouches === 0) {
                isInteracting.value = false;
            } else {
                activeFingers.value = e.numberOfTouches;
            }
        })
        .onFinalize(() => {
            isInteracting.value = false;
            activeFingers.value = 0;
            fingerX1.value = -100;
        }), [width]);

    // Top-level animated styles
    const cursor1Style = useAnimatedStyle(() => ({
        left: fingerX1.value,
        opacity: withTiming(isInteracting.value ? 1 : 0, { duration: 150 })
    }));

    const cursor2Style = useAnimatedStyle(() => ({
        left: fingerX2.value,
        opacity: withTiming((isInteracting.value && activeFingers.value >= 2) ? 1 : 0, { duration: 150 })
    }));

    const rangeHighlightStyle = useAnimatedStyle(() => ({
        left: Math.min(fingerX1.value, fingerX2.value),
        width: Math.abs(fingerX1.value - fingerX2.value),
        overflow: 'hidden',
        opacity: withTiming((isInteracting.value && activeFingers.value >= 2) ? 1 : 0, { duration: 150 })
    }));

    const glowTranslateStyle = useAnimatedStyle(() => ({
        transform: [{ translateX: -Math.min(fingerX1.value, fingerX2.value) }]
    }));

    if (data.length < 2) {
        return (
            <View style={{ height, width, alignItems: 'center', justifyContent: 'center' }}>
                <ActivityIndicator color={accentColor} />
                <Text style={{ color: '#666', fontSize: 10, fontWeight: 'bold', marginTop: 8 }}>Loading Chart...</Text>
            </View>
        );
    }

    const pathColor = graphState.isInteracting ? (graphState.isPositive ? '#4ADE80' : '#F87171') : accentColor;
    const highlightColor = graphState.isPositive ? '#4ADE80' : '#F87171';

    return (
        <GestureDetector gesture={gesture}>
            <View style={{ width, height }}>
                <LineChart.Provider data={data}>
                    <LineChart height={height} width={width}>
                        <LineChart.Path color={pathColor} width={2} />

                        {/* Overlays */}
                        <View style={StyleSheet.absoluteFill} pointerEvents="none">
                            {/* Glowing Segment for 2-finger interaction */}
                            <Animated.View style={[
                                { position: 'absolute', top: 0, bottom: 0 },
                                rangeHighlightStyle
                            ]}>
                                <Animated.View style={[{ width, height }, glowTranslateStyle]}>
                                    <LineChart.Path color={highlightColor} width={4}>
                                        <LineChart.Gradient color={highlightColor} />
                                    </LineChart.Path>
                                    <LineChart.Path color="rgba(255,255,255,0.5)" width={1} />
                                </Animated.View>
                            </Animated.View>

                            {/* Cursors */}
                            <Animated.View style={[
                                { position: 'absolute', top: 0, bottom: 0, width: 2, backgroundColor: 'white' },
                                cursor1Style
                            ]} />
                            <Animated.View style={[
                                { position: 'absolute', top: 0, bottom: 0, width: 2, backgroundColor: 'white' },
                                cursor2Style
                            ]} />
                        </View>
                    </LineChart>
                </LineChart.Provider>
            </View>
        </GestureDetector>
    );
};
