export default {
  rules: {
    'block-no-empty': true,
    'color-hex-length': 'short',
    'color-named': 'never',
    'color-no-hex': true,
    'declaration-no-important': true,
    'declaration-property-value-disallowed-list': [
      {
        '/^(?:column-gap|gap|margin|padding|row-gap)/': [
          /\b\d+(?:\.\d+)?(?:px|rem)\b/u,
        ],
      },
      { message: 'Spacing must use an Ant Token CSS variable.' },
    ],
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
