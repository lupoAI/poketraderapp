import { Platform } from 'react-native';

// When testing on a physical device, 'localhost' points to the device itself.
// We must use the machine's local IP address instead.
const LOCAL_IP = '192.168.8.194';

export const API_URL = Platform.OS === 'android' ? `http://10.0.2.2:8001` : `http://${LOCAL_IP}:8001`;
