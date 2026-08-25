import type { ReactNode } from 'react';
import {
  Alert,
  Card,
  Collapse,
  Descriptions,
  Empty,
  Image,
  Skeleton,
  Space,
  Tag,
  ThoughtChain,
  Typography,
  Video,
} from '../../ui/antd';
import './media.css';

export type ArtifactStatus = 'queued' | 'processing' | 'done' | 'failed';

export interface VideoArtifactProps {
  url?: string | null;
  title?: string;
  loading?: boolean;
  error?: string;
  onMediaError?: () => void;
}

interface VideoArtifactCardProps extends VideoArtifactProps {
  fallbackTitle: string;
  emptyDescription: string;
}

function VideoArtifactCard({
  url,
  title,
  loading = false,
  error,
  onMediaError,
  fallbackTitle,
  emptyDescription,
}: VideoArtifactCardProps) {
  const resolvedTitle = title ?? fallbackTitle;

  return (
    <Card title={resolvedTitle}>
      {loading ? (
        <div aria-label={`正在加载${resolvedTitle}`}>
          <Skeleton active paragraph={{ rows: 4 }} />
        </div>
      ) : error ? (
        <Alert type="error" showIcon title={error} />
      ) : url ? (
        <Video
          className="media-artifact-video"
          label={resolvedTitle}
          src={url}
          controls
          playsInline
          preload="metadata"
          onError={onMediaError}
        />
      ) : (
        <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={emptyDescription} />
      )}
    </Card>
  );
}

export function SourceVideo(props: VideoArtifactProps) {
  return (
    <VideoArtifactCard
      {...props}
      fallbackTitle="源视频"
      emptyDescription="暂无源视频"
    />
  );
}

export function FinalVideo(props: VideoArtifactProps) {
  return (
    <VideoArtifactCard
      {...props}
      fallbackTitle="最终成片"
      emptyDescription="暂无最终成片"
    />
  );
}

export interface DialogueArtifact {
  id: string;
  text: string;
  startTime?: string;
  endTime?: string;
}

export interface ArtifactSummaryProps {
  duration?: string | number;
  keyframeCount?: number;
  dialogue?: readonly DialogueArtifact[];
  loading?: boolean;
  error?: string;
}

function dialogueTime(dialogue: DialogueArtifact) {
  if (dialogue.startTime && dialogue.endTime) {
    return `${dialogue.startTime} — ${dialogue.endTime}`;
  }

  return dialogue.startTime ?? dialogue.endTime;
}

export function ArtifactSummary({
  duration,
  keyframeCount,
  dialogue,
  loading = false,
  error,
}: ArtifactSummaryProps) {
  if (loading) {
    return (
      <div aria-label="正在加载分析产物摘要">
        <Skeleton active />
      </div>
    );
  }

  if (error) {
    return <Alert type="error" showIcon title={error} />;
  }

  const metrics = [
    duration !== undefined
      ? { key: 'duration', label: '时长', children: typeof duration === 'number' ? `${duration} 秒` : duration }
      : undefined,
    keyframeCount !== undefined
      ? { key: 'keyframes', label: '关键帧', children: `${keyframeCount} 帧` }
      : undefined,
    dialogue !== undefined
      ? { key: 'dialogue', label: '台词', children: `${dialogue.length} 段` }
      : undefined,
  ].filter((item): item is NonNullable<typeof item> => item !== undefined);

  if (metrics.length === 0) {
    return <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无分析产物" />;
  }

  return (
    <Card title="分析产物摘要">
      <Descriptions column={{ xs: 1, sm: 3 }} items={metrics} />
      {dialogue && dialogue.length > 0 ? (
        <Space orientation="vertical" className="media-artifact-list" aria-label="识别台词">
          {dialogue.map((item) => (
            <Space orientation="vertical" size="small" key={item.id}>
              {dialogueTime(item) ? (
                <Typography.Text type="secondary">{dialogueTime(item)}</Typography.Text>
              ) : null}
              <Typography.Text>{item.text}</Typography.Text>
            </Space>
          ))}
        </Space>
      ) : null}
    </Card>
  );
}

export interface KeyframeArtifact {
  id: string;
  url?: string;
  alt: string;
  timestamp?: string;
  error?: string;
}

export interface KeyframeGalleryProps {
  keyframes: readonly KeyframeArtifact[];
  loading?: boolean;
  error?: string;
  emptyDescription?: ReactNode;
}

export function KeyframeGallery({
  keyframes,
  loading = false,
  error,
  emptyDescription = '暂无关键帧',
}: KeyframeGalleryProps) {
  if (loading) {
    return (
      <div aria-label="正在加载关键帧">
        <Skeleton active />
      </div>
    );
  }

  if (error) {
    return <Alert type="error" showIcon title={error} />;
  }

  if (keyframes.length === 0) {
    return <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={emptyDescription} />;
  }

  return (
    <Image.PreviewGroup>
      <div className="keyframe-gallery">
        {keyframes.map((keyframe) => (
          <figure className="keyframe-gallery-item" key={keyframe.id}>
            {keyframe.error ? (
              <Alert type="error" showIcon title={keyframe.error} />
            ) : keyframe.url ? (
              <Image src={keyframe.url} alt={keyframe.alt} />
            ) : (
              <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="关键帧地址缺失" />
            )}
            {keyframe.timestamp ? <figcaption>{keyframe.timestamp}</figcaption> : null}
          </figure>
        ))}
      </div>
    </Image.PreviewGroup>
  );
}

export interface SegmentArtifact {
  id: string;
  title?: string;
  status: ArtifactStatus;
  startTime?: string;
  endTime?: string;
  firstFrame?: KeyframeArtifact;
  lastFrame?: KeyframeArtifact;
  prompt?: string;
  dialogue?: readonly string[];
  loading?: boolean;
  error?: string;
}

export interface LongVideoSegmentsProps {
  segments: readonly SegmentArtifact[];
  loading?: boolean;
  error?: string;
  defaultExpandedSegmentIds?: readonly string[];
}

const segmentStatusPresentation: Record<
  ArtifactStatus,
  { label: string; chainStatus?: 'loading' | 'success' | 'error' }
> = {
  queued: { label: '排队中' },
  processing: { label: '处理中', chainStatus: 'loading' },
  done: { label: '已完成', chainStatus: 'success' },
  failed: { label: '失败', chainStatus: 'error' },
};

function segmentTime(segment: SegmentArtifact) {
  if (segment.startTime && segment.endTime) {
    return `${segment.startTime} — ${segment.endTime}`;
  }

  return segment.startTime ?? segment.endTime;
}

function SegmentContent({ segment, title }: { segment: SegmentArtifact; title: string }) {
  if (segment.loading) {
    return (
      <div aria-label={`正在加载${title}`}>
        <Skeleton active />
      </div>
    );
  }

  if (segment.error) {
    return <Alert type="error" showIcon title={segment.error} />;
  }

  const frames = [segment.firstFrame, segment.lastFrame].filter(
    (frame): frame is KeyframeArtifact => frame !== undefined,
  );

  return (
    <Collapse
      size="small"
      items={[
        {
          key: 'frames',
          label: '首尾帧',
          children: <KeyframeGallery keyframes={frames} emptyDescription="暂无首尾帧" />,
        },
        {
          key: 'prompt',
          label: '提示词',
          children: segment.prompt ? (
            <Typography.Paragraph copyable>{segment.prompt}</Typography.Paragraph>
          ) : (
            <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无提示词" />
          ),
        },
        {
          key: 'dialogue',
          label: '台词',
          children: segment.dialogue && segment.dialogue.length > 0 ? (
            <Space orientation="vertical" className="media-artifact-list">
              {segment.dialogue.map((line, index) => (
                <Typography.Text key={`${segment.id}-dialogue-${index}`}>{line}</Typography.Text>
              ))}
            </Space>
          ) : (
            <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无台词" />
          ),
        },
      ]}
    />
  );
}

export function LongVideoSegments({
  segments,
  loading = false,
  error,
  defaultExpandedSegmentIds = [],
}: LongVideoSegmentsProps) {
  if (loading) {
    return (
      <div aria-label="正在加载长视频分段">
        <Skeleton active paragraph={{ rows: 6 }} />
      </div>
    );
  }

  if (error) {
    return <Alert type="error" showIcon title={error} />;
  }

  if (segments.length === 0) {
    return <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无长视频分段" />;
  }

  return (
    <Card title="长视频分段" extra={<Tag>共 {segments.length} 个分段</Tag>}>
      <ThoughtChain
        defaultExpandedKeys={[...defaultExpandedSegmentIds]}
        items={segments.map((segment, index) => {
          const presentation = segmentStatusPresentation[segment.status];
          const title = segment.title ?? `分段 ${index + 1}`;
          return {
            key: segment.id,
            title,
            description: (
              <Space size="small">
                <Tag>{presentation.label}</Tag>
                {segmentTime(segment) ? (
                  <Typography.Text type="secondary">{segmentTime(segment)}</Typography.Text>
                ) : null}
              </Space>
            ),
            status: presentation.chainStatus,
            collapsible: true,
            destroyOnHidden: false,
            content: <SegmentContent segment={segment} title={title} />,
          };
        })}
      />
    </Card>
  );
}
