import { act, fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import App from './App';

describe('Duet AI prototype', () => {
  const clickLabeledControl = async (user: ReturnType<typeof userEvent.setup>, label: string) => {
    const input = screen.getByLabelText(label);
    await user.click(input.closest('label') ?? input);
  };

  beforeEach(() => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it('switches preset conversations and creates a new local conversation', async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    render(<App />);

    expect(screen.getByText('分析完成待生成')).toBeInTheDocument();
    await user.click(screen.getByText('长视频分片生成中'));
    expect(screen.getByText('正在生成 4 个视频分片')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: '新建会话' }));
    expect(screen.getByText('开始一个新的视频任务')).toBeInTheDocument();
  });

  it('gates submission and simulates queued to processing to done', async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    render(<App />);
    await user.click(screen.getByRole('button', { name: '新建会话' }));

    const analyze = screen.getByRole('button', { name: '开始分析' });
    expect(analyze).toBeDisabled();
    await user.type(screen.getByLabelText('视频链接'), 'https://example.com/demo.mp4');
    expect(analyze).toBeEnabled();
    await user.click(analyze);
    expect(screen.getByText('任务已进入队列')).toBeInTheDocument();

    act(() => vi.advanceTimersByTime(900));
    expect(screen.getByText('正在理解视频内容')).toBeInTheDocument();
    act(() => vi.advanceTimersByTime(1800));
    expect(screen.getByText('视频分析完成')).toBeInTheDocument();
  });

  it('collects transcript handling during creation and makes analysis transcript read-only', async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    render(<App />);
    await user.click(screen.getByRole('button', { name: '新建会话' }));

    expect(screen.getByRole('radiogroup', { name: '创建阶段台词处理' })).toBeInTheDocument();
    await clickLabeledControl(user, '创建台词 翻译为');
    expect(screen.getByLabelText('创建阶段目标语言')).toBeInTheDocument();

    await user.type(screen.getByLabelText('视频链接'), 'https://example.com/demo.mp4');
    await user.click(screen.getByRole('button', { name: '开始分析' }));
    act(() => vi.advanceTimersByTime(900));
    act(() => vi.advanceTimersByTime(1800));

    expect(screen.queryByRole('radiogroup', { name: '创建阶段台词处理' })).not.toBeInTheDocument();
    expect(screen.getByRole('heading', { name: '识别台词' })).toBeInTheDocument();
  });

  it('locks generation parameters and completes segmented generation', async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    render(<App />);

    await clickLabeledControl(user, '画幅 9:16');
    await clickLabeledControl(user, '分辨率 768p');
    await clickLabeledControl(user, '适配方式 留白');
    await clickLabeledControl(user, 'H3 台词模式 自定义台词');
    await user.clear(screen.getByLabelText('自定义台词内容'));
    await user.type(screen.getByLabelText('自定义台词内容'), '从清晨第一杯咖啡开始。');
    await user.click(screen.getByRole('button', { name: '生成视频' }));

    expect(screen.getByLabelText('画幅 9:16')).toBeDisabled();
    expect(screen.getByLabelText('H3 台词模式 自定义台词')).toBeDisabled();
    expect(screen.getByLabelText('自定义台词内容')).toBeDisabled();
    expect(screen.getByRole('button', { name: '生成中' })).toBeDisabled();
    act(() => vi.advanceTimersByTime(5000));
    expect(screen.getByText('全部分片生成完成')).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: '已冻结生成参数' })).toBeInTheDocument();
    expect(screen.getByDisplayValue('从清晨第一杯咖啡开始。')).toBeDisabled();
    expect(screen.getByText('最终成片已就绪')).toBeInTheDocument();
  });

  it('runs keyframe post processing locally with current product options', async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    render(<App />);
    await user.click(screen.getByRole('button', { name: '打开关键帧后处理' }));

    expect(screen.getByRole('dialog', { name: '关键帧后处理' })).toBeInTheDocument();
    expect(screen.getByLabelText('去字幕水印')).toBeInTheDocument();
    await user.click(screen.getByLabelText('去版权物品'));
    expect(screen.queryByLabelText('智能字幕')).not.toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: '开始后处理' }));
    expect(screen.getByRole('button', { name: '处理中' })).toBeDisabled();
    act(() => vi.advanceTimersByTime(2200));
    expect(screen.getByText('后处理已完成')).toBeInTheDocument();
  });

  it('shows frozen parameters on preset completion without a final-card post action', async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    render(<App />);
    await user.click(screen.getByText('最终成片完成'));

    expect(screen.getByRole('heading', { name: '已冻结生成参数' })).toBeInTheDocument();
    expect(screen.getByLabelText('画幅 16:9')).toBeDisabled();
    expect(screen.getByLabelText('H3 台词模式 自动台词')).toBeDisabled();
    expect(screen.queryByRole('button', { name: '打开后处理' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: '打开关键帧后处理' })).toBeInTheDocument();
  });

  it('never calls fetch while using local interactions', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch');
    const xhrOpenSpy = vi.spyOn(XMLHttpRequest.prototype, 'open');
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    render(<App />);
    await user.click(screen.getByText('长视频分片生成中'));
    await user.click(screen.getByText('最终成片完成'));
    await user.click(screen.getByRole('button', { name: '新建会话' }));
    await user.click(screen.getByText('上传文件'));
    const fileInput = screen.getByLabelText('选择视频文件');
    fireEvent.change(fileInput, {
      target: { files: [new File(['local-only'], 'demo.mp4', { type: 'video/mp4' })] },
    });
    expect(screen.getByRole('button', { name: '开始分析' })).toBeEnabled();
    expect(fetchSpy).not.toHaveBeenCalled();
    expect(xhrOpenSpy).not.toHaveBeenCalled();
  });
});
