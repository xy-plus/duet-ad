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
        '/^(?:height|max-height|min-height)$/': [
          /^(?!100dvh$).*dvh.*$/u,
        ],
        '/^(?:inline-size|max-inline-size|max-width|min-inline-size|min-width|width)$/': [
          /\b(?:5[7-9]|[6-9]\d|\d{3,})(?:\.\d+)?rem\b/u,
        ],
      },
      { message: 'Token-driven properties must use an Ant Token CSS variable.' },
    ],
    'function-disallowed-list': [
      'color',
      /^drop-shadow$/iu,
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
  overrides: [
    {
      files: ['src/app/**/*.css', 'src/features/**/*.css'],
      rules: {
        'property-disallowed-list': [
          [
            /^--/u,
            /^(?:background(?:-.+)?|border(?:-.+)?|box-shadow|color|filter|font(?:-.+)?|outline(?:-.+)?|text-shadow)$/u,
          ],
          {
            message: 'Feature CSS is geometry-only; use AntD/AntD X components and theme Tokens for visual surfaces.',
          },
        ],
      },
    },
  ],
};
