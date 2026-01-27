
import React from 'react';
import { View, Text, Modal, TouchableOpacity } from 'react-native';
import { BlurView } from 'expo-blur';
import Ionicons from '@expo/vector-icons/Ionicons';

interface SelectionOption {
    label: string;
    value: string;
}

interface SelectionMenuProps {
    visible: boolean;
    onClose: () => void;
    title: string;
    current: string;
    onSelect: (value: any) => void;
    options: SelectionOption[];
}

export const SelectionMenu: React.FC<SelectionMenuProps> = ({
    visible,
    onClose,
    title,
    current,
    onSelect,
    options
}) => {
    return (
        <Modal
            visible={visible}
            transparent
            animationType="fade"
            onRequestClose={onClose}
        >
            <TouchableOpacity
                activeOpacity={1}
                onPress={onClose}
                style={{ flex: 1, backgroundColor: 'rgba(0,0,0,0.7)', justifyContent: 'flex-end' }}
            >
                <BlurView intensity={80} tint="dark" style={{ borderTopLeftRadius: 32, borderTopRightRadius: 32, overflow: 'hidden' }}>
                    <View className="p-8 pb-12">
                        <View className="flex-row justify-between items-center mb-6">
                            <Text className="text-white text-2xl font-bold">{title}</Text>
                            <TouchableOpacity onPress={onClose} className="bg-white/10 p-2 rounded-full">
                                <Ionicons name="close" size={20} color="white" />
                            </TouchableOpacity>
                        </View>
                        <View className="gap-3">
                            {options.map((opt) => (
                                <TouchableOpacity
                                    key={opt.value}
                                    onPress={() => { onSelect(opt.value); onClose(); }}
                                    className={`flex-row items-center justify-between p-5 rounded-3xl border ${current === opt.value ? 'bg-yellow-400 border-yellow-400' : 'bg-white/5 border-white/10'}`}
                                >
                                    <Text className={`font-bold text-lg ${current === opt.value ? 'text-black' : 'text-white'}`}>{opt.label}</Text>
                                    {current === opt.value && <Ionicons name="checkmark-circle" size={24} color="black" />}
                                </TouchableOpacity>
                            ))}
                        </View>
                    </View>
                </BlurView>
            </TouchableOpacity>
        </Modal>
    );
};
