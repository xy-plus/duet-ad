import { useReducer } from 'react';

export type SourceMode = 'link' | 'upload';
export type TranscriptMode = 'keep' | 'rewrite' | 'translate';
export type H3DialogueMode = 'auto' | 'edit' | 'custom' | 'none';
export type AnalysisStatus = 'idle' | 'queued' | 'processing' | 'done';
export type GenerationStatus = 'idle' | 'running' | 'succeeded';
export type PostStatus = 'idle' | 'running' | 'succeeded';
export type Phase = 'draft' | 'analysisDone' | 'generating' | 'complete';

export interface Conversation {
  id: string;
  title: string;
  group: '今天' | '最近 7 天';
  phase: Phase;
  sourceMode: SourceMode;
  sourceUrl: string;
  uploadFileName: string;
  analysisStatus: AnalysisStatus;
  analysisTicks: number;
  transcriptMode: TranscriptMode;
  targetLanguage: string;
  h3DialogueMode: H3DialogueMode;
  h3Dialogue: string;
  prompt: string;
  aspect: '16:9' | '9:16';
  resolution: '480p' | '768p';
  fit: 'crop' | 'pad';
  generationStatus: GenerationStatus;
  segmentsDone: number;
  simulatingGeneration: boolean;
  postStatus: PostStatus;
  postTicks: number;
}

export interface MockState {
  activeId: string;
  nextId: number;
  conversations: Conversation[];
}

type Action =
  | { type: 'select'; id: string }
  | { type: 'new' }
  | { type: 'sourceMode'; mode: SourceMode }
  | { type: 'sourceUrl'; value: string }
  | { type: 'file'; name: string }
  | { type: 'submitAnalysis' }
  | { type: 'transcriptMode'; mode: TranscriptMode }
  | { type: 'targetLanguage'; language: string }
  | { type: 'h3DialogueMode'; mode: H3DialogueMode }
  | { type: 'h3Dialogue'; value: string }
  | { type: 'prompt'; prompt: string }
  | { type: 'aspect'; value: Conversation['aspect'] }
  | { type: 'resolution'; value: Conversation['resolution'] }
  | { type: 'fit'; value: Conversation['fit'] }
  | { type: 'startGeneration' }
  | { type: 'startPost' }
  | { type: 'tick' };

const shared = {
  sourceMode: 'upload' as const,
  sourceUrl: '',
  analysisStatus: 'done' as const,
  analysisTicks: 0,
  transcriptMode: 'keep' as const,
  targetLanguage: 'English',
  h3DialogueMode: 'auto' as const,
  h3Dialogue: '一杯好咖啡，让城市的早晨慢下来。今天，也给自己留一点从容。',
  prompt:
    '以原视频的镜头节奏为基准，保留咖啡杯与城市晨光的视觉锚点，生成自然、克制、具有真实摄影质感的新品短片。',
  aspect: '16:9' as const,
  resolution: '480p' as const,
  fit: 'crop' as const,
  simulatingGeneration: false,
  postStatus: 'idle' as const,
  postTicks: 0,
};

const initialState: MockState = {
  activeId: 'analysis-ready',
  nextId: 1,
  conversations: [
    {
      ...shared,
      id: 'analysis-ready',
      title: '分析完成待生成',
      group: '今天',
      phase: 'analysisDone',
      uploadFileName: '城市咖啡新品短片.mp4',
      generationStatus: 'idle',
      segmentsDone: 0,
    },
    {
      ...shared,
      id: 'segment-running',
      title: '长视频分片生成中',
      group: '今天',
      phase: 'generating',
      uploadFileName: '城市漫游长片.mp4',
      aspect: '9:16',
      resolution: '768p',
      fit: 'pad',
      generationStatus: 'running',
      segmentsDone: 2,
    },
    {
      ...shared,
      id: 'final-ready',
      title: '最终成片完成',
      group: '最近 7 天',
      phase: 'complete',
      uploadFileName: '秋日护肤品牌片.mp4',
      generationStatus: 'succeeded',
      segmentsDone: 4,
    },
  ],
};

function updateActive(
  state: MockState,
  updater: (item: Conversation) => Conversation,
): MockState {
  return {
    ...state,
    conversations: state.conversations.map((item) =>
      item.id === state.activeId ? updater(item) : item,
    ),
  };
}

export function mockReducer(state: MockState, action: Action): MockState {
  switch (action.type) {
    case 'select':
      return state.conversations.some((item) => item.id === action.id)
        ? { ...state, activeId: action.id }
        : state;
    case 'new': {
      const id = `local-${state.nextId}`;
      const conversation: Conversation = {
        ...shared,
        id,
        title: `新视频任务 ${state.nextId}`,
        group: '今天',
        phase: 'draft',
        sourceMode: 'link',
        sourceUrl: '',
        uploadFileName: '',
        analysisStatus: 'idle',
        analysisTicks: 0,
        prompt: '',
        generationStatus: 'idle',
        segmentsDone: 0,
      };
      return {
        activeId: id,
        nextId: state.nextId + 1,
        conversations: [conversation, ...state.conversations],
      };
    }
    case 'sourceMode':
      return updateActive(state, (item) =>
        item.phase === 'draft' && item.analysisStatus === 'idle'
          ? { ...item, sourceMode: action.mode }
          : item,
      );
    case 'sourceUrl':
      return updateActive(state, (item) =>
        item.phase === 'draft' && item.analysisStatus === 'idle'
          ? { ...item, sourceUrl: action.value }
          : item,
      );
    case 'file':
      return updateActive(state, (item) =>
        item.phase === 'draft' && item.analysisStatus === 'idle'
          ? { ...item, uploadFileName: action.name }
          : item,
      );
    case 'submitAnalysis':
      return updateActive(state, (item) => {
        const selectedSource = item.sourceMode === 'link' ? item.sourceUrl : item.uploadFileName;
        if (item.analysisStatus !== 'idle' || !selectedSource.trim()) return item;
        return { ...item, analysisStatus: 'queued', analysisTicks: 0 };
      });
    case 'transcriptMode':
      return updateActive(state, (item) =>
        item.phase === 'draft' && item.analysisStatus === 'idle'
          ? { ...item, transcriptMode: action.mode }
          : item,
      );
    case 'targetLanguage':
      return updateActive(state, (item) =>
        item.phase === 'draft' && item.analysisStatus === 'idle'
          ? { ...item, targetLanguage: action.language }
          : item,
      );
    case 'h3DialogueMode':
      return updateActive(state, (item) =>
        item.phase === 'analysisDone' && item.generationStatus === 'idle'
          ? { ...item, h3DialogueMode: action.mode }
          : item,
      );
    case 'h3Dialogue':
      return updateActive(state, (item) =>
        item.phase === 'analysisDone' && item.generationStatus === 'idle'
          ? { ...item, h3Dialogue: action.value }
          : item,
      );
    case 'prompt':
      return updateActive(state, (item) =>
        item.phase === 'analysisDone' && item.generationStatus === 'idle'
          ? { ...item, prompt: action.prompt }
          : item,
      );
    case 'aspect':
      return updateActive(state, (item) =>
        item.generationStatus === 'idle' ? { ...item, aspect: action.value } : item,
      );
    case 'resolution':
      return updateActive(state, (item) =>
        item.generationStatus === 'idle' ? { ...item, resolution: action.value } : item,
      );
    case 'fit':
      return updateActive(state, (item) =>
        item.generationStatus === 'idle' ? { ...item, fit: action.value } : item,
      );
    case 'startGeneration':
      return updateActive(state, (item) => {
        if (item.phase !== 'analysisDone' || item.generationStatus !== 'idle') return item;
        if (
          (item.h3DialogueMode === 'edit' || item.h3DialogueMode === 'custom') &&
          !item.h3Dialogue.trim()
        ) return item;
        return {
          ...item,
          phase: 'generating',
          generationStatus: 'running',
          segmentsDone: 0,
          simulatingGeneration: true,
        };
      });
    case 'startPost':
      return updateActive(state, (item) => {
        if (item.phase === 'draft' || item.analysisStatus !== 'done' || item.postStatus !== 'idle') return item;
        return { ...item, postStatus: 'running', postTicks: 0 };
      });
    case 'tick':
      {
        let changed = false;
        const conversations = state.conversations.map((item) => {
          let next = item;

          if (next.analysisStatus === 'queued') {
            next = { ...next, analysisStatus: 'processing', analysisTicks: 1 };
          } else if (next.analysisStatus === 'processing') {
            const analysisTicks = next.analysisTicks + 1;
            next = analysisTicks >= 3
              ? {
                  ...next,
                  phase: 'analysisDone',
                  analysisStatus: 'done',
                  analysisTicks,
                  title: '本地视频分析',
                  prompt: shared.prompt,
                }
              : { ...next, analysisTicks };
          }

          if (next.generationStatus === 'running' && next.simulatingGeneration) {
            const segmentsDone = Math.min(next.segmentsDone + 1, 4);
            next = segmentsDone === 4
              ? {
                  ...next,
                  phase: 'complete',
                  generationStatus: 'succeeded',
                  segmentsDone,
                  simulatingGeneration: false,
                }
              : { ...next, segmentsDone };
          }

          if (next.postStatus === 'running') {
            const postTicks = next.postTicks + 1;
            next = postTicks >= 3
              ? { ...next, postStatus: 'succeeded', postTicks }
              : { ...next, postTicks };
          }

          changed ||= next !== item;
          return next;
        });
        return changed ? { ...state, conversations } : state;
      }
  }
}

export function getInitialMockState(): MockState {
  return {
    ...initialState,
    conversations: initialState.conversations.map((item) => ({ ...item })),
  };
}

export function useMockStore() {
  const [state, dispatch] = useReducer(mockReducer, undefined, getInitialMockState);
  const active = state.conversations.find((item) => item.id === state.activeId)!;
  return { state, active, dispatch };
}

export type MockDispatch = React.Dispatch<Action>;
