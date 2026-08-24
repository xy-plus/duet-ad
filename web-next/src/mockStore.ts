import { useReducer } from 'react';

export type SourceMode = 'link' | 'upload';
export type TranscriptMode = 'keep' | 'rewrite' | 'translate';
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
  sourceValue: string;
  fileName?: string;
  analysisStatus: AnalysisStatus;
  transcriptMode: TranscriptMode;
  targetLanguage: string;
  prompt: string;
  aspect: '16:9' | '9:16';
  resolution: '480p' | '768p';
  fit: 'crop' | 'pad';
  generationStatus: GenerationStatus;
  segmentsDone: number;
  simulatingGeneration: boolean;
  postStatus: PostStatus;
}

interface State {
  activeId: string;
  nextId: number;
  conversations: Conversation[];
}

type Action =
  | { type: 'select'; id: string }
  | { type: 'new' }
  | { type: 'sourceMode'; mode: SourceMode }
  | { type: 'sourceValue'; value: string }
  | { type: 'file'; name: string }
  | { type: 'submitAnalysis' }
  | { type: 'analysisProcessing'; id: string }
  | { type: 'analysisDone'; id: string }
  | { type: 'transcriptMode'; mode: TranscriptMode }
  | { type: 'targetLanguage'; language: string }
  | { type: 'prompt'; prompt: string }
  | { type: 'aspect'; value: Conversation['aspect'] }
  | { type: 'resolution'; value: Conversation['resolution'] }
  | { type: 'fit'; value: Conversation['fit'] }
  | { type: 'startGeneration' }
  | { type: 'generationTick'; id: string }
  | { type: 'startPost' }
  | { type: 'postDone'; id: string };

const shared = {
  sourceMode: 'link' as const,
  analysisStatus: 'done' as const,
  transcriptMode: 'keep' as const,
  targetLanguage: 'English',
  prompt:
    '以原视频的镜头节奏为基准，保留咖啡杯与城市晨光的视觉锚点，生成自然、克制、具有真实摄影质感的新品短片。',
  aspect: '16:9' as const,
  resolution: '480p' as const,
  fit: 'crop' as const,
  simulatingGeneration: false,
  postStatus: 'idle' as const,
};

const initialState: State = {
  activeId: 'analysis-ready',
  nextId: 1,
  conversations: [
    {
      ...shared,
      id: 'analysis-ready',
      title: '分析完成待生成',
      group: '今天',
      phase: 'analysisDone',
      sourceValue: '城市咖啡新品短片.mp4',
      generationStatus: 'idle',
      segmentsDone: 0,
    },
    {
      ...shared,
      id: 'segment-running',
      title: '长视频分片生成中',
      group: '今天',
      phase: 'generating',
      sourceValue: '城市漫游长片.mp4',
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
      sourceValue: '秋日护肤品牌片.mp4',
      generationStatus: 'succeeded',
      segmentsDone: 4,
    },
  ],
};

function updateActive(state: State, updater: (item: Conversation) => Conversation): State {
  return {
    ...state,
    conversations: state.conversations.map((item) =>
      item.id === state.activeId ? updater(item) : item,
    ),
  };
}

function updateById(
  state: State,
  id: string,
  updater: (item: Conversation) => Conversation,
): State {
  return {
    ...state,
    conversations: state.conversations.map((item) => (item.id === id ? updater(item) : item)),
  };
}

function reducer(state: State, action: Action): State {
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
        sourceValue: '',
        analysisStatus: 'idle',
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
      return updateActive(state, (item) => ({
        ...item,
        sourceMode: action.mode,
        sourceValue: action.mode === 'link' ? item.sourceValue : '',
        fileName: action.mode === 'upload' ? item.fileName : undefined,
      }));
    case 'sourceValue':
      return updateActive(state, (item) => ({ ...item, sourceValue: action.value }));
    case 'file':
      return updateActive(state, (item) => ({ ...item, fileName: action.name, sourceValue: action.name }));
    case 'submitAnalysis':
      return updateActive(state, (item) => {
        if (item.analysisStatus !== 'idle' || !item.sourceValue.trim()) return item;
        return { ...item, analysisStatus: 'queued' };
      });
    case 'analysisProcessing':
      return updateById(state, action.id, (item) =>
        item.analysisStatus === 'queued' ? { ...item, analysisStatus: 'processing' } : item,
      );
    case 'analysisDone':
      return updateById(state, action.id, (item) =>
        item.analysisStatus === 'processing'
          ? {
              ...item,
              phase: 'analysisDone',
              analysisStatus: 'done',
              title: '本地视频分析',
              prompt: shared.prompt,
            }
          : item,
      );
    case 'transcriptMode':
      return updateActive(state, (item) => ({ ...item, transcriptMode: action.mode }));
    case 'targetLanguage':
      return updateActive(state, (item) => ({ ...item, targetLanguage: action.language }));
    case 'prompt':
      return updateActive(state, (item) => ({ ...item, prompt: action.prompt }));
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
        return {
          ...item,
          phase: 'generating',
          generationStatus: 'running',
          segmentsDone: 0,
          simulatingGeneration: true,
        };
      });
    case 'generationTick':
      return updateById(state, action.id, (item) => {
        if (item.generationStatus !== 'running' || !item.simulatingGeneration) return item;
        const segmentsDone = Math.min(item.segmentsDone + 1, 4);
        if (segmentsDone === 4) {
          return {
            ...item,
            phase: 'complete',
            generationStatus: 'succeeded',
            segmentsDone,
            simulatingGeneration: false,
          };
        }
        return { ...item, segmentsDone };
      });
    case 'startPost':
      return updateActive(state, (item) => {
        if (item.phase !== 'complete' || item.postStatus !== 'idle') return item;
        return { ...item, postStatus: 'running' };
      });
    case 'postDone':
      return updateById(state, action.id, (item) =>
        item.postStatus === 'running' ? { ...item, postStatus: 'succeeded' } : item,
      );
  }
}

export function useMockStore() {
  const [state, dispatch] = useReducer(reducer, initialState);
  const active = state.conversations.find((item) => item.id === state.activeId)!;
  return { state, active, dispatch };
}

export type MockDispatch = React.Dispatch<Action>;
