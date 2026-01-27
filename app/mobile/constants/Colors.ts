const tintColorLight = '#2f95dc';
const tintColorDark = '#FFCB05'; // Poketrader Yellow

export default {
  light: {
    text: '#000',
    background: '#fff',
    tint: tintColorLight,
    tabIconDefault: '#ccc',
    tabIconSelected: tintColorLight,
  },
  dark: {
    text: '#fff',
    background: '#000000',
    tint: '#FFCB05', // Poketrader Yellow
    tabIconDefault: '#8F8F8F',
    tabIconSelected: '#FFCB05',
    cardBackground: '#111111',
    cardBorder: '#222222',
    accentYellow: '#FFCB05',
    accentGold: '#D5A100',
  },
};
export const TYPE_COLORS: Record<string, string> = {
  // TCG Specific Names
  Grass: '#7AC74C',
  Fire: '#EE8130',
  Water: '#6390F0',
  Lightning: '#F7D02C',
  Psychic: '#F95587',
  Fighting: '#C22E28',
  Darkness: '#705746',
  Metal: '#B7B7CE',
  Fairy: '#D685AD',
  Dragon: '#6F35FC',
  Colorless: '#A8A77A',

  // Standard Game Names
  Normal: '#A8A77A',
  Electric: '#F7D02C',
  Ice: '#96D9D6',
  Poison: '#A33EA1',
  Ground: '#E2BF65',
  Flying: '#A98FF3',
  Bug: '#A6B91A',
  Rock: '#B6A136',
  Ghost: '#735797',
  Dark: '#705746',
  Steel: '#B7B7CE',
};
