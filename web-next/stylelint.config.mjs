export default {
  reportDisables: true,
  rules: {
    'block-no-empty': true,
    'color-hex-length': 'short',
    'color-named': 'never',
    'color-no-hex': true,
    'comment-word-disallowed-list': ['stylelint-disable'],
    'declaration-no-important': true,
    'declaration-property-value-disallowed-list': [
      {
        '/^(?:column-gap|gap|margin|padding|row-gap)/': [
          /\b\d+(?:\.\d+)?(?:px|rem)\b/u,
        ],
        '/^(?:border(?:-[a-z]+){0,3}-radius|box-shadow|text-shadow|font(?:-family|-size)?)$/': [
          /^(?!var\(--ant-[\w-]+\)$).+/u,
        ],
        '/^filter$/': [
          /drop-shadow\s*\(/iu,
        ],
      },
      { message: 'Token-driven properties must use an Ant Token CSS variable.' },
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
