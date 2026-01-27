import { StyleSheet, TouchableOpacity, ScrollView } from 'react-native';
import { Text, View } from '@/components/Themed';
import { SafeAreaView } from 'react-native-safe-area-context';
import FontAwesome from '@expo/vector-icons/FontAwesome';

export default function AnalyticsScreen() {
    return (
        <SafeAreaView className="flex-1 bg-black" edges={['top']}>
            <ScrollView className="flex-1 px-6">
                <View className="py-4 mb-6">
                    <TouchableOpacity className="w-10 h-10 bg-slate-900 rounded-full items-center justify-center mb-6">
                        <FontAwesome name="arrow-left" size={16} color="white" />
                    </TouchableOpacity>
                    <Text className="text-4xl font-bold text-white">Analytics</Text>
                </View>

                <View className="flex-row justify-between items-center mb-6">
                    <View className="flex-row items-center">
                        <Text className="text-white text-lg font-bold mr-2">Personal</Text>
                        <FontAwesome name="chevron-down" size={12} color="white" />
                    </View>
                    <Text className="text-blue-500 font-bold">This month</Text>
                </View>

                {/* Spent Chart Stub */}
                <View className="bg-slate-900 p-6 rounded-3xl mb-4 border border-slate-800">
                    <Text className="text-gray-500 text-xs font-bold uppercase mb-2">Spent</Text>
                    <View className="flex-row items-center mb-4">
                        <Text className="text-3xl font-bold text-white mr-2">£4,417</Text>
                        <FontAwesome name="caret-down" size={14} color="#4ade80" />
                        <Text className="text-green-400 font-bold ml-1">£40</Text>
                    </View>
                    <View className="h-20 w-full justify-end">
                        <View className="h-0.5 bg-gray-700 w-full" />
                        <View className="h-0.5 bg-gray-500 w-3/4 absolute bottom-0" />
                        <View className="w-3 h-3 bg-green-500 rounded-full absolute bottom-[-5] left-0 shadow-lg shadow-green-500" />
                    </View>
                </View>

                <View className="flex-row gap-4 mb-8">
                    <View className="flex-1 bg-slate-900 p-6 rounded-3xl border border-slate-800">
                        <Text className="text-gray-500 text-xs font-bold uppercase mb-2">Income</Text>
                        <Text className="text-2xl font-bold text-white">£0</Text>
                    </View>
                    <View className="flex-1 bg-slate-900 p-6 rounded-3xl border border-slate-800">
                        <Text className="text-gray-500 text-xs font-bold uppercase mb-2">Net cash flow</Text>
                        <Text className="text-2xl font-bold text-white mb-2">-£4,417</Text>
                        <View className="flex-row items-center">
                            <FontAwesome name="minus-circle" size={12} color="#f87171" />
                            <Text className="text-red-400 text-xs font-bold ml-1">Negative</Text>
                        </View>
                    </View>
                </View>

                <View className="mb-6">
                    <Text className="text-2xl font-bold text-white mb-6">Overview</Text>
                    <View className="bg-slate-900 p-6 rounded-3xl border border-slate-800">
                        <Text className="text-gray-500 text-xs font-bold uppercase mb-2">Total assets</Text>
                        <Text className="text-3xl font-bold text-white mb-8">£15,044</Text>
                        <View className="h-10 w-full border-b border-gray-700 justify-end">
                            {/* Simulated chart line */}
                            <View className="h-full w-full border-green-500 border-b-2 opacity-50" />
                        </View>
                    </View>
                </View>
            </ScrollView>
        </SafeAreaView>
    );
}
