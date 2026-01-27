import React from 'react';
import Ionicons from '@expo/vector-icons/Ionicons';
import { Tabs } from 'expo-router';
import { Platform, StyleSheet, View, TouchableOpacity } from 'react-native';
import { BlurView } from 'expo-blur';

function TabBarIcon(props: {
  name: React.ComponentProps<typeof Ionicons>['name'];
  color: string;
}) {
  return <Ionicons size={22} {...props} />;
}

// Custom Camera Button Component
const ScanButton = ({ children, onPress }: any) => (
  // Use TouchableOpacity for simplicity across platforms, or import TouchableNativeFeedback correctly
  <TouchableOpacity onPress={onPress} activeOpacity={0.8}>
    <View
      style={{
        justifyContent: 'center',
        alignItems: 'center',
        width: 64,
        height: 64,
        backgroundColor: '#D5A100', // Premium Yellow
        borderRadius: 32,
        shadowColor: '#D5A100',
        shadowOffset: { width: 0, height: 8 },
        shadowOpacity: 0.4,
        shadowRadius: 10,
        elevation: 10,
        marginLeft: 10, // Push it away from the bar
      }}
    >
      <Ionicons name="scan" size={30} color="black" />
    </View>
  </TouchableOpacity>
);

export default function TabLayout() {
  return (
    <Tabs
      screenOptions={{
        tabBarActiveTintColor: '#D5A100', // Premium Gold
        tabBarInactiveTintColor: '#8E8E93',
        headerShown: false,
        tabBarLabelStyle: { fontSize: 10, fontWeight: '700', marginBottom: 5 }, // Reduced margin
        tabBarStyle: {
          position: 'absolute',
          backgroundColor: 'transparent',
          borderTopWidth: 0,
          bottom: 24,
          left: 0,
          right: 0,
          marginHorizontal: 20,
          height: 64,
          elevation: 0,
          justifyContent: 'center',
          alignItems: 'center',
        },
        tabBarItemStyle: {
          height: 64,
          justifyContent: 'center',
          alignItems: 'center',
          paddingTop: 0,
        },
        tabBarBackground: () => (
          // Split Background: Only covers the first 3 tabs (approx 75% width)
          <BlurView
            intensity={80}
            tint="dark"
            style={{
              position: 'absolute',
              left: 0,
              bottom: 0,
              top: 0,
              width: '78%', // Covers Collection, Sets, Trends
              borderRadius: 32,
              borderWidth: 1,
              borderColor: 'rgba(255,255,255,0.1)',
              overflow: 'hidden',
            }}
          />
        ),
      }}>
      <Tabs.Screen
        name="index"
        options={{
          title: 'Collection',
          tabBarIcon: ({ color }) => <TabBarIcon name="layers" color={color} />,
          tabBarItemStyle: { paddingTop: 10, height: 64 }, // Apply padding locally
        }}
      />
      <Tabs.Screen
        name="market"
        options={{
          title: 'Sets',
          tabBarIcon: ({ color }) => <TabBarIcon name="grid" color={color} />,
          tabBarItemStyle: { paddingTop: 10, height: 64 }, // Apply padding locally
        }}
      />
      <Tabs.Screen
        name="trends"
        options={{
          title: 'Trends',
          tabBarIcon: ({ color }) => <TabBarIcon name="stats-chart" color={color} />,
          tabBarItemStyle: { paddingTop: 10, height: 64 }, // Apply padding locally
        }}
      />
      <Tabs.Screen
        name="scan"
        options={{
          title: '', // Remove title
          tabBarIcon: ({ color }) => <TabBarIcon name="scan" color={color} />,
          tabBarButton: (props) => <ScanButton {...props} />, // Use custom floating button
          tabBarStyle: { display: 'none' }, // Hide tab bar (and button) when on this screen
        }}
      />
    </Tabs>
  );
}
