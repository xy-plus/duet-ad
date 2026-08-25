import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  ArtifactSummary,
  FinalVideo,
  KeyframeGallery,
  LongVideoSegments,
  SourceVideo,
  type SegmentArtifact,
} from './MediaArtifacts';

afterEach(cleanup);

const segments: SegmentArtifact[] = Array.from({ length: 3 }, (_, index) => ({
  id: `segment-${index + 1}`,
  title: `分段 ${index + 1}`,
  status: index === 0 ? 'done' : index === 1 ? 'processing' : 'queued',
  startTime: `${index * 10}s`,
  endTime: `${(index + 1) * 10}s`,
  firstFrame: {
    id: `first-${index + 1}`,
    url: `/frames/${index + 1}-first.jpg`,
    alt: `分段 ${index + 1} 首帧`,
  },
  lastFrame: {
    id: `last-${index + 1}`,
    url: `/frames/${index + 1}-last.jpg`,
    alt: `分段 ${index + 1} 尾帧`,
  },
  prompt: `分段 ${index + 1} 的提示词`,
  dialogue: [`分段 ${index + 1} 的台词`],
}));

describe('authenticated media artifacts', () => {
  it('keeps source and final URLs controlled by props and updates the video element', () => {
    const { rerender } = render(
      <SourceVideo url="https://media.test/source-a.mp4" title="源素材" />,
    );

    const source = screen.getByLabelText('源素材');
    expect(source).toHaveAttribute('controls');
    expect(source).toHaveAttribute('src', 'https://media.test/source-a.mp4');
    expect(source.closest('.ant-card')).not.toBeNull();

    rerender(<SourceVideo url="https://media.test/source-b.mp4" title="源素材" />);
    expect(screen.getByLabelText('源素材')).toHaveAttribute(
      'src',
      'https://media.test/source-b.mp4',
    );

    rerender(<FinalVideo url="https://media.test/final.mp4" title="最终成片" />);
    expect(screen.getByLabelText('最终成片')).toHaveAttribute(
      'src',
      'https://media.test/final.mp4',
    );
    expect(screen.getByLabelText('最终成片').closest('.ant-card')).not.toBeNull();
  });

  it('shows media loading, missing and failure states instead of a fake player', () => {
    const { rerender } = render(<SourceVideo loading />);
    expect(screen.getByLabelText('正在加载源视频')).toBeInTheDocument();

    rerender(<SourceVideo error="鉴权链接已失效" />);
    expect(screen.getByRole('alert')).toHaveTextContent('鉴权链接已失效');
    expect(screen.queryByLabelText('源视频')).not.toBeInTheDocument();

    rerender(<FinalVideo />);
    expect(screen.getByText('暂无最终成片')).toBeInTheDocument();
  });

  it('delegates native playback failure so the owner can replace the URL or error prop', () => {
    const onMediaError = vi.fn();
    render(
      <SourceVideo
        url="https://media.test/expired.mp4"
        onMediaError={onMediaError}
      />,
    );

    fireEvent.error(screen.getByLabelText('源视频'));
    expect(onMediaError).toHaveBeenCalledOnce();
  });
});

describe('ArtifactSummary and keyframes', () => {
  it('renders only metrics present in API data and never invents codec, rhythm or shot count', () => {
    render(
      <ArtifactSummary
        duration="38 秒"
        keyframeCount={2}
        dialogue={[{ id: 'd1', text: '真实台词' }]}
      />,
    );

    expect(screen.getByText('38 秒')).toBeInTheDocument();
    expect(screen.getByText('2 帧')).toBeInTheDocument();
    expect(screen.getByText('真实台词')).toBeInTheDocument();
    expect(screen.queryByText(/codec|编码|节奏|镜头数/i)).not.toBeInTheDocument();
  });

  it('renders prop-provided keyframes in a preview group and exposes item failures', async () => {
    const user = userEvent.setup();
    const { rerender } = render(
      <KeyframeGallery
        keyframes={[
          { id: 'k1', url: '/frames/a.jpg', alt: '雨夜首帧', timestamp: '00:01' },
          { id: 'k2', url: '/frames/b.jpg', alt: '清晨尾帧', timestamp: '00:09' },
        ]}
      />,
    );

    expect(screen.getByAltText('雨夜首帧')).toHaveAttribute('src', '/frames/a.jpg');
    expect(screen.getByAltText('清晨尾帧')).toHaveAttribute('src', '/frames/b.jpg');
    await user.click(screen.getByAltText('雨夜首帧'));
    await waitFor(() => {
      expect(screen.getByRole('dialog')).toHaveClass('ant-image-preview');
    });

    rerender(
      <KeyframeGallery
        keyframes={[{ id: 'broken', alt: '损坏帧', error: '关键帧加载失败' }]}
      />,
    );
    expect(screen.getByRole('alert')).toHaveTextContent('关键帧加载失败');
  });
});

describe('LongVideoSegments', () => {
  it('renders the actual dynamic segment count and collapsible frame, prompt and dialogue evidence', async () => {
    const user = userEvent.setup();
    render(<LongVideoSegments segments={segments} defaultExpandedSegmentIds={['segment-1']} />);

    expect(screen.getByText('共 3 个分段')).toBeInTheDocument();
    expect(screen.getByText('分段 3')).toBeInTheDocument();
    expect(screen.queryByText('分段 1 的提示词')).not.toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: '提示词' }));
    expect(screen.getByText('分段 1 的提示词')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: '首尾帧' }));
    expect(screen.getByAltText('分段 1 首帧')).toBeInTheDocument();
    expect(screen.getByAltText('分段 1 尾帧')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: '台词' }));
    expect(screen.getByText('分段 1 的台词')).toBeInTheDocument();
  });

  it('shows segment-level loading and error evidence', () => {
    render(
      <LongVideoSegments
        segments={[
          { id: 'loading', title: '加载分段', status: 'processing', loading: true },
          { id: 'failed', title: '失败分段', status: 'failed', error: '分段产物读取失败' },
        ]}
        defaultExpandedSegmentIds={['loading', 'failed']}
      />,
    );

    expect(screen.getByLabelText('正在加载加载分段')).toBeInTheDocument();
    expect(screen.getByRole('alert')).toHaveTextContent('分段产物读取失败');
  });
});
