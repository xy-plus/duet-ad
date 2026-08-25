export type CreateConversationSource =
  | { readonly kind: 'file'; readonly file: File }
  | { readonly kind: 'url'; readonly url: string };

export type CreateConversationVoice =
  | { readonly mode: 'keep' }
  | { readonly mode: 'rewrite' }
  | { readonly mode: 'translate'; readonly targetLanguage: string };

export interface CreateConversationInput {
  readonly source: CreateConversationSource;
  readonly note?: string;
  readonly voice?: CreateConversationVoice;
}

export interface CreateConversationIntent {
  readonly source: CreateConversationSource;
  readonly note: string;
  readonly voice: CreateConversationVoice;
  readonly clientRequestId: string;
}

const CLIENT_REQUEST_ID = /^[0-9A-Za-z-]{8,64}$/u;

export function newClientRequestId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  return `rid-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 12)}`;
}

function freezeSource(source: CreateConversationSource): CreateConversationSource {
  if (source.kind === 'file') {
    if (!source.file || typeof source.file.name !== 'string') {
      throw new Error('请选择视频文件');
    }
    return Object.freeze({ kind: 'file', file: source.file });
  }
  if (source.kind === 'url') {
    const url = String(source.url ?? '').trim();
    if (!url) throw new Error('请填写视频链接');
    return Object.freeze({ kind: 'url', url });
  }
  throw new Error('视频来源无效');
}

function freezeVoice(voice: CreateConversationVoice | undefined): CreateConversationVoice {
  if (!voice || voice.mode === 'keep') return Object.freeze({ mode: 'keep' });
  if (voice.mode === 'rewrite') return Object.freeze({ mode: 'rewrite' });
  if (voice.mode === 'translate') {
    const targetLanguage = String(voice.targetLanguage ?? '').trim();
    if (!targetLanguage) throw new Error('请填写翻译目标语言');
    return Object.freeze({ mode: 'translate', targetLanguage });
  }
  throw new Error('口播处理模式无效');
}

export function createConversationIntent(
  input: CreateConversationInput,
  requestIdFactory: () => string = newClientRequestId,
): CreateConversationIntent {
  const clientRequestId = String(requestIdFactory()).trim();
  if (!CLIENT_REQUEST_ID.test(clientRequestId)) throw new Error('生成请求标识无效');

  return Object.freeze({
    source: freezeSource(input.source),
    note: String(input.note ?? '').trim(),
    voice: freezeVoice(input.voice),
    clientRequestId,
  });
}
