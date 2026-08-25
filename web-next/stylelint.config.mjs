export default {
  rules: {
    'block-no-empty': true,
    'color-hex-length': 'short',
    'color-named': 'never',
    'color-no-hex': true,
    'declaration-no-important': true,
    'function-disallowed-list': [
      'color',
      'hsl',
      'hsla',
      'hwb',
      'lab',
      'lch',
      'oklab',
      'oklch',
      'rgb',
      'rgba',
    ],
    'selector-disallowed-list': [/\.ant-/u],
  },
};
