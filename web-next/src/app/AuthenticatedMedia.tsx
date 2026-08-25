import type { ReactNode } from 'react';
import type { ApiClient } from '../api';
import type { ConversationDetail, ConversationSegment } from '../domain';
import { useAuthenticatedFileUrl } from '../state';
import {
  Alert,
  Card,
  Empty,
  Image,
  Skeleton,
  Typography,
} from '../ui/antd';
import {
  FinalVideo,
  LongVideoSegments,
  SourceVideo,
  type ArtifactStatus,
  type KeyframeArtifact,
} from '../features/media';
import './app.css';

function message(error: unknown, fallback: string): string {
  return error instanceof Error ? error.message : fallback;
}

interface AuthenticatedVideoProps {
  apiClient: ApiClient;
  conversationId: string;
  fileName: 'source.mp4' | 'generated.mp4';
}

export function AuthenticatedVideo({
  apiClient,
  conversationId,
  fileName,
}: AuthenticatedVideoProps) {
  const file = useAuthenticatedFileUrl(apiClient, conversationId, fileName);
  const props = {
    url: file.url,
    loading: file.isPending,
    error: file.error ? message(file.error, '视频加载失败') : undefined,
  };
  return fileName === 'source.mp4' ? <SourceVideo {...props} /> : <FinalVideo {...props} />;
}

interface AuthenticatedImageProps {
  apiClient: ApiClient;
  alt: string;
  conversationId: string;
  fileName: string;
  caption?: ReactNode;
}

export function AuthenticatedImage({
  apiClient,
  alt,
  conversationId,
  fileName,
  caption,
}: AuthenticatedImageProps) {
  const file = useAuthenticatedFileUrl(apiClient, conversationId, fileName);
  return (
    <figure className="app-media-grid__item">
      {file.isPending ? (
        <div aria-label={`正在加载${alt}`}><Skeleton.Image active /></div>
      ) : file.error ? (
        <Alert type="error" showIcon title={message(file.error, `${alt}加载失败`)} />
      ) : file.url ? (
        <Image src={file.url} alt={alt} />
      ) : (
        <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={`${alt}地址缺失`} />
      )}
      {caption ? <figcaption><Typography.Text type="secondary">{caption}</Typography.Text></figcaption> : null}
    </figure>
  );
}

interface AuthenticatedImageGridProps {
  apiClient: ApiClient;
  conversationId: string;
  files: readonly { readonly fileName: string; readonly alt: string; readonly caption?: ReactNode }[];
  title?: ReactNode;
}

export function AuthenticatedImageGrid({
  apiClient,
  conversationId,
  files,
  title,
}: AuthenticatedImageGridProps) {
  if (files.length === 0) return null;
  const content = (
    <Image.PreviewGroup>
      <div className="app-media-grid">
        {files.map((file) => (
          <AuthenticatedImage
            {...file}
            apiClient={apiClient}
            conversationId={conversationId}
            key={file.fileName}
          />
        ))}
      </div>
    </Image.PreviewGroup>
  );
  return title ? <Card title={title}>{content}</Card> : content;
}

function artifactStatus(detail: ConversationDetail, segment: ConversationSegment): ArtifactStatus {
  const generation = detail.generation?.segments?.find(({ index }) => index === segment.index);
  if (generation?.status === 'succeeded') return 'done';
  if (generation?.status === 'failed' || generation?.status === 'submission_unknown') return 'failed';
  if (generation?.status === 'running' || generation?.status === 'submitting') return 'processing';
  if (generation) return 'queued';
  if (detail.status === 'done') return 'done';
  if (detail.status === 'failed') return 'failed';
  return detail.status === 'processing' ? 'processing' : 'queued';
}

function frameArtifact(
  name: string | undefined,
  alt: string,
  query: ReturnType<typeof useAuthenticatedFileUrl>,
): KeyframeArtifact | undefined {
  if (!name) return undefined;
  return {
    id: name,
    alt,
    url: query.url ?? undefined,
    error: query.error ? message(query.error, `${alt}加载失败`) : undefined,
  };
}

interface AuthenticatedSegmentProps {
  apiClient: ApiClient;
  detail: ConversationDetail;
  segment: ConversationSegment;
}

function AuthenticatedSegment({ apiClient, detail, segment }: AuthenticatedSegmentProps) {
  const names = segment.keyframes ?? [];
  const firstName = names[0];
  const lastName = names.length > 1 ? names[names.length - 1] : undefined;
  const prefix = `segments/${segment.index}/work/keyframes`;
  const first = useAuthenticatedFileUrl(
    apiClient,
    detail.id,
    firstName ? `${prefix}/${firstName}` : null,
  );
  const last = useAuthenticatedFileUrl(
    apiClient,
    detail.id,
    lastName ? `${prefix}/${lastName}` : null,
  );
  const generation = detail.generation?.segments?.find(({ index }) => index === segment.index);

  return (
    <div className="app-detail-stack">
      <LongVideoSegments
        defaultExpandedSegmentIds={[`${detail.id}-segment-${segment.index}`]}
        segments={[{
          id: `${detail.id}-segment-${segment.index}`,
          title: `第 ${segment.index} 段`,
          status: artifactStatus(detail, segment),
          startTime: typeof segment.start_s === 'number' ? `${segment.start_s} 秒` : undefined,
          endTime: typeof segment.end_s === 'number' ? `${segment.end_s} 秒` : undefined,
          firstFrame: frameArtifact(firstName, `第 ${segment.index} 段首帧`, first),
          lastFrame: frameArtifact(lastName, `第 ${segment.index} 段尾帧`, last),
          prompt: segment.prompt ?? undefined,
          dialogue: segment.lines,
          loading: Boolean((firstName && first.isPending) || (lastName && last.isPending)),
          error: generation?.error ?? undefined,
        }]}
      />
      {names.length > 2 ? (
        <AuthenticatedImageGrid
          apiClient={apiClient}
          conversationId={detail.id}
          files={names.slice(1, -1).map((name, index) => ({
            fileName: `${prefix}/${name}`,
            alt: `第 ${segment.index} 段关键帧 ${index + 2}`,
          }))}
          title={`第 ${segment.index} 段其他关键帧`}
        />
      ) : null}
    </div>
  );
}

export function AuthenticatedSegments({
  apiClient,
  detail,
}: {
  apiClient: ApiClient;
  detail: ConversationDetail;
}) {
  if (detail.segments.length === 0) return null;
  return (
    <section aria-label="长视频分段" className="app-detail-stack">
      {detail.segments.map((segment) => (
        <AuthenticatedSegment
          apiClient={apiClient}
          detail={detail}
          key={segment.index}
          segment={segment}
        />
      ))}
    </section>
  );
}
