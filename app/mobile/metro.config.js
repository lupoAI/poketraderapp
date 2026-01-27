const { getDefaultConfig } = require("expo/metro-config");
const { withNativeWind } = require("nativewind/metro");

const config = getDefaultConfig(__dirname);

// Explicitly add .ts and .tsx to source extensions for Skia resolution
config.resolver.sourceExts = [...config.resolver.sourceExts, "ts", "tsx"];

module.exports = withNativeWind(config, { input: "./global.css" });
