import { useEffect, useState } from 'react';
import {
  App as AntApp,
  Button,
  Checkbox,
  Collapse,
  ConfigProvider,
  Drawer,
  Image,
  Input,
  Modal,
  Progress,
  Radio,
  Segmented,
  Select,
  Skeleton,
  Space,
  Tag,
  Typography,
} from 'antd';
import {
  CheckCircleFilled,
  ClockCircleOutlined,
  EditOutlined,
  MenuOutlined,
  ProductOutlined,
  ScissorOutlined,
  VideoCameraOutlined,
  VideoCameraAddOutlined,
} from '@ant-design/icons';
import { Attachments, Bubble, Conversations, Sender, ThoughtChain } from '@ant-design/x';
import { appTheme } from './theme';
import {
  type Conversation,
  type H3DialogueMode,
  type MockDispatch,
  type SourceMode,
  type TranscriptMode,
  useMockStore,
} from './mockStore';

const { Text, Title, Paragraph } = Typography;
const languageOptions = [
  { value: 'English', label: '英语' },
  { value: 'Japanese', label: '日语' },
  { value: 'Korean', label: '韩语' },
  { value: 'Spanish', label: '西班牙语' },
];

function StatusBadge({ item }: { item: Conversation }) {
  if (item.postStatus === 'running') return <Tag icon={<ClockCircleOutlined />}>后处理中</Tag>;
  if (item.postStatus === 'succeeded') return <Tag color="success">后处理完成</Tag>;
  if (item.phase === 'complete') return <Tag color="success">成片完成</Tag>;
  if (item.generationStatus === 'running') return <Tag icon={<ClockCircleOutlined />}>生成中</Tag>;
  if (item.analysisStatus === 'done') return <Tag>分析完成</Tag>;
  if (item.analysisStatus === 'processing') return <Tag icon={<ClockCircleOutlined />}>分析中</Tag>;
  if (item.analysisStatus === 'queued') return <Tag>排队中</Tag>;
  return <Tag>未开始</Tag>;
}

interface SidebarProps {
  conversations: Conversation[];
  activeId: string;
  onSelect: (id: string) => void;
  onNew: () => void;
}

function ConversationSidebar({ conversations, activeId, onSelect, onNew }: SidebarProps) {
  const items = conversations.map((item) => ({
    key: item.id,
    label: item.title,
    group: item.group,
    icon:
      item.phase === 'complete' ? (
        <CheckCircleFilled className="conversation-icon-success" />
      ) : item.generationStatus === 'running' ? (
        <ClockCircleOutlined />
      ) : (
        <VideoCameraOutlined />
      ),
  }));

  return (
    <div className="sidebar-inner">
      <div className="brand-block" aria-label="Duet AI 视频工作台">
        <span className="brand-mark"><ProductOutlined /></span>
        <div>
          <Text strong>Duet AI</Text>
          <Text type="secondary" className="brand-subtitle">视频工作台</Text>
        </div>
      </div>
      <Conversations
        className="conversation-list"
        items={items}
        activeKey={activeId}
        onActiveChange={(key) => onSelect(String(key))}
        groupable={{
          label: (group) => <span className="conversation-group">{group}</span>,
        }}
        creation={{
          label: '新建会话',
          icon: <span aria-hidden="true" className="creation-symbol">＋</span>,
          onClick: onNew,
        }}
      />
      <div className="sidebar-footnote">
        <Text type="secondary">本地交互原型 · 刷新后复位</Text>
      </div>
    </div>
  );
}

function SourceVideo({ name }: { name: string }) {
  return (
    <div className="media-card media-card-source">
      <div className="media-visual">
        <span className="media-kicker">SOURCE VIDEO</span>
        <VideoCameraOutlined className="media-play-icon" />
        <span className="media-duration">00:32</span>
      </div>
      <div className="media-meta">
        <div>
          <Text strong>{name || '待分析视频'}</Text>
          <Text type="secondary" className="media-description">1920 × 1080 · H.264 · 本地占位</Text>
        </div>
        <Tag variant="filled">源视频</Tag>
      </div>
    </div>
  );
}

interface AnalysisSummaryProps {
  item: Conversation;
  dispatch: MockDispatch;
  onPost: () => void;
}

function AnalysisSummary({ item, dispatch, onPost }: AnalysisSummaryProps) {
  const [editingPrompt, setEditingPrompt] = useState(false);
  const [draftPrompt, setDraftPrompt] = useState(item.prompt);
  const promptLocked = item.generationStatus !== 'idle';

  useEffect(() => {
    setDraftPrompt(item.prompt);
    setEditingPrompt(false);
  }, [item.id, item.prompt, item.generationStatus]);

  const savePrompt = () => {
    const nextPrompt = draftPrompt.trim();
    if (!nextPrompt || promptLocked) return;
    dispatch({ type: 'prompt', prompt: nextPrompt });
    setEditingPrompt(false);
  };

  return (
    <div className="assistant-content">
      <div className="result-heading">
        <div>
          <Title level={3}>视频分析完成</Title>
          <Paragraph type="secondary">已识别叙事结构、关键画面和口播内容，可以继续调整生成策略。</Paragraph>
        </div>
        <Tag color="success" icon={<CheckCircleFilled />}>可生成</Tag>
      </div>

      <div className="metric-strip" aria-label="视频分析摘要">
        <div><Text type="secondary">时长</Text><Text strong>00:32</Text></div>
        <div><Text type="secondary">镜头</Text><Text strong>4 个</Text></div>
        <div><Text type="secondary">口播</Text><Text strong>2 段</Text></div>
        <div><Text type="secondary">节奏</Text><Text strong>舒缓</Text></div>
      </div>

      <section className="content-section" aria-labelledby="keyframe-title">
        <div className="section-heading">
          <div>
            <Title level={5} id="keyframe-title">关键帧</Title>
            <Text type="secondary">从转场和主体变化中选取</Text>
          </div>
          <Space>
            <Tag variant="filled">4 帧</Tag>
            <Button
              aria-label="打开关键帧后处理"
              icon={<ScissorOutlined />}
              onClick={onPost}
            >
              关键帧后处理
            </Button>
          </Space>
        </div>
        <Image.PreviewGroup>
          <div className="keyframe-grid">
            {[1, 2, 3, 4].map((frame) => (
              <figure key={frame} className="keyframe-item">
                <Image src={`/placeholders/keyframe-${frame}.svg`} alt={`关键帧 ${frame}`} />
                <figcaption>00:{String(frame * 7).padStart(2, '0')}</figcaption>
              </figure>
            ))}
          </div>
        </Image.PreviewGroup>
      </section>

      <section className="content-section" aria-labelledby="dialogue-title">
        <div className="section-heading">
          <div>
            <Title level={5} id="dialogue-title">识别台词</Title>
            <Text type="secondary">分析阶段的只读识别结果，可在 H3 生成设置中引用或编辑。</Text>
          </div>
          <Tag variant="filled">
            {item.transcriptMode === 'keep' && '原文保持'}
            {item.transcriptMode === 'rewrite' && '原文改编'}
            {item.transcriptMode === 'translate' &&
              `翻译为 ${languageOptions.find(({ value }) => value === item.targetLanguage)?.label ?? item.targetLanguage}`}
          </Tag>
        </div>
        <div className="transcript-preview">
          <Text type="secondary">00:05 — 00:10</Text>
          <Paragraph>一杯好咖啡，让城市的早晨慢下来。</Paragraph>
          <Text type="secondary">00:21 — 00:27</Text>
          <Paragraph>今天，也给自己留一点从容。</Paragraph>
        </div>
      </section>

      <Collapse
        ghost
        className="prompt-collapse"
        items={[
          {
            key: 'prompt',
            label: <Text strong>生成提示词</Text>,
            children: editingPrompt && !promptLocked ? (
              <Space orientation="vertical" className="full-width">
                <Input.TextArea
                  aria-label="生成提示词内容"
                  value={draftPrompt}
                  autoSize={{ minRows: 4, maxRows: 8 }}
                  onChange={(event) => setDraftPrompt(event.target.value)}
                />
                <Space>
                  <Button type="primary" onClick={savePrompt} disabled={!draftPrompt.trim()}>保存提示词</Button>
                  <Button onClick={() => { setDraftPrompt(item.prompt); setEditingPrompt(false); }}>取消</Button>
                </Space>
              </Space>
            ) : (
              <div className="prompt-readonly">
                <Paragraph>{item.prompt}</Paragraph>
                {!promptLocked && (
                  <Button
                    aria-label="编辑提示词"
                    icon={<EditOutlined />}
                    onClick={() => setEditingPrompt(true)}
                  >
                    编辑提示词
                  </Button>
                )}
              </div>
            ),
          },
        ]}
      />
    </div>
  );
}

interface GenerationPanelProps {
  item: Conversation;
  dispatch: MockDispatch;
  summary?: boolean;
}

function GenerationPanel({ item, dispatch, summary = false }: GenerationPanelProps) {
  const locked = summary || item.generationStatus !== 'idle';
  const dialogueEditable = item.h3DialogueMode === 'edit' || item.h3DialogueMode === 'custom';
  const dialogueMissing = dialogueEditable && !item.h3Dialogue.trim();
  const dialogueInputLabel = item.h3DialogueMode === 'edit' ? '识别台词内容' : '自定义台词内容';

  return (
    <section className="generation-panel" aria-labelledby="generation-title">
      <div className="section-heading">
        <div>
          <Title level={4} id="generation-title">{summary ? '已冻结生成参数' : '生成设置'}</Title>
          <Text type="secondary">
            {summary ? '本地冻结参数快照，不可修改。' : '提交后参数锁定；原型会在本地模拟 4 个分片。'}
          </Text>
        </div>
        <Tag variant="filled">{summary ? '不可修改' : '预计 4 段'}</Tag>
      </div>
      <div className="parameter-grid">
        <div className="parameter-field">
          <Text strong>画幅</Text>
          <Radio.Group
            value={item.aspect}
            disabled={locked}
            onChange={(event) => dispatch({ type: 'aspect', value: event.target.value })}
          >
            <Radio.Button value="16:9" aria-label="画幅 16:9">16:9</Radio.Button>
            <Radio.Button value="9:16" aria-label="画幅 9:16">9:16</Radio.Button>
          </Radio.Group>
        </div>
        <div className="parameter-field">
          <Text strong>分辨率</Text>
          <Radio.Group
            value={item.resolution}
            disabled={locked}
            onChange={(event) => dispatch({ type: 'resolution', value: event.target.value })}
          >
            <Radio.Button value="480p" aria-label="分辨率 480p">480p</Radio.Button>
            <Radio.Button value="768p" aria-label="分辨率 768p">768p</Radio.Button>
          </Radio.Group>
        </div>
        <div className="parameter-field">
          <Text strong>画面适配</Text>
          <Radio.Group
            value={item.fit}
            disabled={locked}
            onChange={(event) => dispatch({ type: 'fit', value: event.target.value })}
          >
            <Radio.Button value="crop" aria-label="适配方式 裁切">裁切铺满</Radio.Button>
            <Radio.Button value="pad" aria-label="适配方式 留白">完整留白</Radio.Button>
          </Radio.Group>
        </div>
        <div className="parameter-field parameter-dialogue">
          <Text strong>H3 台词模式</Text>
          <Radio.Group
            className="dialogue-mode-group"
            aria-label="H3 台词模式"
            value={item.h3DialogueMode}
            disabled={locked}
            onChange={(event) => dispatch({ type: 'h3DialogueMode', mode: event.target.value as H3DialogueMode })}
          >
            <Radio.Button value="auto" aria-label="H3 台词模式 自动台词">自动台词</Radio.Button>
            <Radio.Button value="edit" aria-label="H3 台词模式 编辑识别台词">编辑识别台词</Radio.Button>
            <Radio.Button value="custom" aria-label="H3 台词模式 自定义台词">自定义台词</Radio.Button>
            <Radio.Button value="none" aria-label="H3 台词模式 无台词">无台词</Radio.Button>
          </Radio.Group>
          {dialogueEditable && (
            <Input.TextArea
              aria-label={dialogueInputLabel}
              value={item.h3Dialogue}
              disabled={locked}
              status={dialogueMissing ? 'error' : undefined}
              rows={3}
              placeholder={item.h3DialogueMode === 'edit' ? '编辑分析阶段识别出的台词' : '输入成片需要使用的自定义台词'}
              onChange={(event) => dispatch({ type: 'h3Dialogue', value: event.target.value })}
            />
          )}
          {item.h3DialogueMode === 'auto' && <Text type="secondary">由 H3 根据画面和生成提示自动组织台词</Text>}
          {item.h3DialogueMode === 'none' && <Text type="secondary">成片不生成或保留台词</Text>}
        </div>
      </div>
      {!summary && (
        <div className="generation-actions">
          <Text type={dialogueMissing ? 'danger' : 'secondary'}>
            {dialogueMissing ? '请先填写台词内容' : '所有任务仅保存在当前浏览器内存中'}
          </Text>
          <Button
            type="primary"
            icon={<ProductOutlined />}
            aria-label={locked ? '参数已锁定' : '生成视频'}
            onClick={() => dispatch({ type: 'startGeneration' })}
            disabled={locked || dialogueMissing}
          >
            {locked ? '参数已锁定' : '生成视频'}
          </Button>
        </div>
      )}
    </section>
  );
}

function GenerationProgress({ item }: { item: Conversation }) {
  const percent = Math.round((item.segmentsDone / 4) * 100);
  const chainItems = Array.from({ length: 4 }, (_, index) => {
    const segment = index + 1;
    const completed = segment <= item.segmentsDone;
    const active = segment === item.segmentsDone + 1;
    return {
      key: `segment-${segment}`,
      title: `分片 ${segment} · 00:${String((segment - 1) * 8).padStart(2, '0')} — 00:${String(segment * 8).padStart(2, '0')}`,
      description: completed ? '已生成并通过一致性检查' : active ? '正在生成画面与动作' : '等待前序分片',
      status: completed ? ('success' as const) : active ? ('loading' as const) : undefined,
    };
  });

  return (
    <section className="generation-progress" aria-live="polite">
      <div className="section-heading">
        <div>
          <Title level={3}>正在生成 4 个视频分片</Title>
          <Text type="secondary">参数已锁定，离开当前会话不会中断本地模拟。</Text>
        </div>
        <Button aria-label="生成中" loading disabled>生成中</Button>
      </div>
      <Progress percent={percent} status="active" />
      <ThoughtChain items={chainItems} />
    </section>
  );
}

function FinalVideo() {
  return (
    <section className="final-result" aria-labelledby="final-title">
      <div className="success-callout">
        <CheckCircleFilled />
        <div>
          <Title level={3}>全部分片生成完成</Title>
          <Text type="secondary">拼接与基础音画同步检查已完成</Text>
        </div>
      </div>
      <div className="media-card media-card-final">
        <div className="media-visual">
          <span className="media-kicker">FINAL VIDEO</span>
          <ProductOutlined className="media-play-icon" />
          <span className="media-duration">00:32</span>
        </div>
        <div className="media-meta">
          <div>
            <Text strong id="final-title">最终成片已就绪</Text>
            <Text type="secondary" className="media-description">4 个分片 · 已拼接 · 本地高质量占位</Text>
          </div>
          <Tag color="success">已完成</Tag>
        </div>
      </div>
    </section>
  );
}

interface TaskSenderProps {
  item: Conversation;
  dispatch: MockDispatch;
}

function TaskSender({ item, dispatch }: TaskSenderProps) {
  const [message, setMessage] = useState('');
  const [localFeedback, setLocalFeedback] = useState('');
  const analysisRunning = item.analysisStatus === 'queued' || item.analysisStatus === 'processing';
  const selectedSource = item.sourceMode === 'link' ? item.sourceUrl : item.uploadFileName;
  const canAnalyze = Boolean(selectedSource.trim()) && item.analysisStatus === 'idle';

  const sourceHeader = item.phase === 'draft' ? (
    <div className="sender-source">
      <div className="sender-source-top">
        <Segmented<SourceMode>
          aria-label="视频来源"
          value={item.sourceMode}
          onChange={(mode) => dispatch({ type: 'sourceMode', mode })}
          options={[
            { label: '链接输入', value: 'link' },
            { label: '上传文件', value: 'upload' },
          ]}
          disabled={analysisRunning}
        />
        <Text type="secondary">不会上传到服务器</Text>
      </div>
      {item.sourceMode === 'link' ? (
        <Input
          aria-label="视频链接"
          value={item.sourceUrl}
          disabled={analysisRunning}
          placeholder="https://example.com/video.mp4"
          prefix={<VideoCameraOutlined />}
          onChange={(event) => dispatch({ type: 'sourceUrl', value: event.target.value })}
        />
      ) : (
        <div className="local-upload">
          <input
            id={`video-file-${item.id}`}
            className="visually-hidden-file"
            type="file"
            accept="video/*"
            aria-label="选择视频文件"
            disabled={analysisRunning}
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) dispatch({ type: 'file', name: file.name });
            }}
          />
          <label className="local-upload-trigger" htmlFor={`video-file-${item.id}`}>
            <VideoCameraAddOutlined aria-hidden="true" />
            选择视频文件
          </label>
          {item.uploadFileName && (
            <div className="selected-attachment">
              <Text type="secondary">已选择：{item.uploadFileName}</Text>
              <Attachments
                items={[{ uid: 'local-file', name: item.uploadFileName, status: 'done' }]}
                disabled={analysisRunning}
                customRequest={() => ({ abort: () => undefined })}
                onRemove={() => {
                  dispatch({ type: 'file', name: '' });
                  return true;
                }}
              >
                <span className="attachments-anchor" aria-hidden="true" />
              </Attachments>
            </div>
          )}
        </div>
      )}
      <div className="creation-transcript-options">
        <div className="creation-transcript-heading">
          <Text strong>分析输入台词</Text>
          <Text type="secondary">随分析任务一并提交</Text>
        </div>
        <div className="creation-transcript-controls">
          <Radio.Group
            aria-label="创建阶段台词处理"
            value={item.transcriptMode}
            disabled={analysisRunning}
            onChange={(event) => dispatch({ type: 'transcriptMode', mode: event.target.value as TranscriptMode })}
          >
            <Radio.Button value="keep" aria-label="创建台词 原文保持">原文保持</Radio.Button>
            <Radio.Button value="rewrite" aria-label="创建台词 原文改编">原文改编</Radio.Button>
            <Radio.Button value="translate" aria-label="创建台词 翻译为">翻译为</Radio.Button>
          </Radio.Group>
          {item.transcriptMode === 'translate' && (
            <Select
              aria-label="创建阶段目标语言"
              value={item.targetLanguage}
              disabled={analysisRunning}
              onChange={(language) => dispatch({ type: 'targetLanguage', language })}
              options={languageOptions}
            />
          )}
        </div>
      </div>
    </div>
  ) : null;

  const submitFollowup = (value: string) => {
    if (!value.trim()) return;
    setLocalFeedback('已记录本地调整建议（不会发送到服务端）');
    setMessage('');
  };

  return (
    <div className="sender-wrap">
      <Sender
        value={message}
        onChange={setMessage}
        onSubmit={submitFollowup}
        loading={analysisRunning}
        onCancel={() => undefined}
        autoSize={false}
        placeholder={item.phase === 'draft' ? '可补充视频用途、受众或风格偏好…' : '继续描述你想调整的内容…'}
        header={sourceHeader}
        suffix={item.phase === 'draft' ? false : undefined}
        footer={item.phase === 'draft' ? (
          <div className="sender-footer">
            <div aria-live="polite">
              {item.analysisStatus === 'queued' && <Text>排队中，请稍候</Text>}
              {item.analysisStatus === 'processing' && <Text>分析处理中</Text>}
              {item.analysisStatus === 'idle' && <Text type="secondary">支持链接或本地视频文件</Text>}
            </div>
            <Button
              type="primary"
              loading={analysisRunning}
              disabled={!canAnalyze}
              onClick={() => dispatch({ type: 'submitAnalysis' })}
            >
              开始分析
            </Button>
          </div>
        ) : undefined}
      />
      <div className="sender-disclaimer" aria-live="polite">
        {localFeedback || '本原型不会发起网络请求，所有状态在刷新后复位。'}
      </div>
    </div>
  );
}

interface PostProcessModalProps {
  item: Conversation;
  open: boolean;
  onClose: () => void;
  dispatch: MockDispatch;
}

function PostProcessModal({ item, open, onClose, dispatch }: PostProcessModalProps) {
  const [options, setOptions] = useState<string[]>(['remove_subtitles']);
  const running = item.postStatus === 'running';
  const succeeded = item.postStatus === 'succeeded';

  return (
    <Modal
      title="关键帧后处理"
      open={open}
      onCancel={onClose}
      keyboard={!running}
      mask={{ closable: !running }}
      closable={!running}
      footer={succeeded ? (
        <Button type="primary" onClick={onClose}>完成</Button>
      ) : (
        <Space>
          <Button onClick={onClose} disabled={running}>取消</Button>
          <Button
            type="primary"
            loading={running}
            aria-label={running ? '处理中' : '开始后处理'}
            disabled={running || options.length === 0}
            onClick={() => dispatch({ type: 'startPost' })}
          >
            {running ? '处理中' : '开始后处理'}
          </Button>
        </Space>
      )}
    >
      {succeeded ? (
        <div className="post-success" aria-live="polite">
          <CheckCircleFilled />
          <Title level={4}>后处理已完成</Title>
          <Text type="secondary">所选关键帧清理步骤已在本地模拟完成。</Text>
        </div>
      ) : (
        <div className="post-options">
          <Paragraph type="secondary">选择关键帧清理步骤。运行期间会锁定操作，避免重复提交。</Paragraph>
          <Checkbox.Group
            value={options}
            disabled={running}
            onChange={(values) => setOptions(values as string[])}
          >
            <Space orientation="vertical">
              <Checkbox value="remove_subtitles">去字幕水印</Checkbox>
              <Checkbox value="remove_copyrighted_objects">去版权物品</Checkbox>
            </Space>
          </Checkbox.Group>
          {running && (
            <div className="post-running" aria-live="polite">
              <Progress percent={64} status="active" />
              <Text>正在处理所选步骤…</Text>
            </div>
          )}
        </div>
      )}
    </Modal>
  );
}

function Workspace() {
  const { state, active, dispatch } = useMockStore();
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [postOpen, setPostOpen] = useState(false);

  useEffect(() => {
    const timer = window.setInterval(() => dispatch({ type: 'tick' }), 700);
    return () => window.clearInterval(timer);
  }, [dispatch]);

  const sidebarProps: SidebarProps = {
    conversations: state.conversations,
    activeId: state.activeId,
    onSelect: (id: string) => {
      dispatch({ type: 'select', id });
      setDrawerOpen(false);
      setPostOpen(false);
    },
    onNew: () => {
      dispatch({ type: 'new' });
      setDrawerOpen(false);
      setPostOpen(false);
    },
  };

  return (
    <div className="app-shell">
      <aside className="desktop-sidebar">
        <ConversationSidebar {...sidebarProps} />
      </aside>

      <Drawer
        title="Duet AI"
        placement="left"
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        className="mobile-drawer"
        destroyOnHidden
      >
        <ConversationSidebar {...sidebarProps} />
      </Drawer>

      <main className="workspace">
        <header className="workspace-header">
          <Button
            className="mobile-menu-button"
            type="text"
            icon={<MenuOutlined />}
            aria-label="打开会话列表"
            onClick={() => setDrawerOpen(true)}
          />
          <div>
            <Text strong>Duet 视频任务</Text>
            <Text type="secondary" className="workspace-subtitle">无后端交互演示</Text>
          </div>
          <StatusBadge item={active} />
        </header>

        <div className="thread-scroll" aria-live="polite">
          <div className="thread-content">
            {active.phase === 'draft' && active.analysisStatus === 'idle' && (
              <div className="welcome-block">
                <span className="welcome-icon"><ProductOutlined /></span>
                <Title level={2}>开始一个新的视频任务</Title>
                <Paragraph type="secondary">粘贴视频链接或选择本地文件，Duet 会模拟完成内容理解、提示词整理和分片生成。</Paragraph>
                <div className="capability-row">
                  <Tag>关键帧理解</Tag>
                  <Tag>台词处理</Tag>
                  <Tag>分片生成</Tag>
                  <Tag>成片后处理</Tag>
                </div>
              </div>
            )}

            {active.phase === 'draft' && active.analysisStatus !== 'idle' && (
              <Bubble
                variant="borderless"
                content={active.analysisStatus === 'queued' ? (
                  <div className="analysis-loading">
                    <ClockCircleOutlined />
                    <div><Title level={4}>任务已进入队列</Title><Text type="secondary">正在准备本地分析流程…</Text></div>
                  </div>
                ) : (
                  <div className="analysis-loading">
                    <Skeleton.Avatar active />
                    <div><Title level={4}>正在理解视频内容</Title><Text type="secondary">提取镜头、台词和视觉锚点…</Text></div>
                  </div>
                )}
              />
            )}

            {active.phase !== 'draft' && (
              <>
                <Bubble
                  placement="end"
                  variant="filled"
                  rootClassName="user-bubble"
                  content={<SourceVideo name={active.sourceMode === 'link' ? active.sourceUrl : active.uploadFileName} />}
                />
                <Bubble
                  placement="start"
                  variant="borderless"
                  rootClassName="assistant-bubble"
                  content={<AnalysisSummary item={active} dispatch={dispatch} onPost={() => setPostOpen(true)} />}
                />
                {active.phase === 'analysisDone' && <GenerationPanel item={active} dispatch={dispatch} />}
                {active.phase === 'generating' && (
                  <>
                    <GenerationPanel item={active} dispatch={dispatch} />
                    <GenerationProgress item={active} />
                  </>
                )}
                {active.phase === 'complete' && (
                  <>
                    <GenerationPanel item={active} dispatch={dispatch} summary />
                    <FinalVideo />
                  </>
                )}
              </>
            )}
          </div>
        </div>

        <TaskSender item={active} dispatch={dispatch} />
      </main>

      <PostProcessModal
        item={active}
        open={postOpen}
        onClose={() => { if (active.postStatus !== 'running') setPostOpen(false); }}
        dispatch={dispatch}
      />
    </div>
  );
}

export default function App() {
  return (
    <ConfigProvider theme={appTheme}>
      <AntApp>
        <Workspace />
      </AntApp>
    </ConfigProvider>
  );
}
