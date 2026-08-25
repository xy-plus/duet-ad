import { createContext, useCallback, useContext, useMemo, useRef, useState, type ReactNode } from 'react';
import { Alert, Button, Modal, Space } from '../ui/antd';

interface DraftHandler { save: () => Promise<void>; discard: () => void }
interface GuardValue {
  register: (id: string, handler: DraftHandler | null) => void;
  run: (action: () => void) => void;
}

const GuardContext = createContext<GuardValue | null>(null);
const fallbackGuard: GuardValue = { register: () => undefined, run: (action) => action() };

export function UnsavedDraftProvider({ children }: { children: ReactNode }) {
  const handlers = useRef(new Map<string, DraftHandler>());
  const [pendingAction, setPendingAction] = useState<(() => void) | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string>();
  const register = useCallback((id: string, handler: DraftHandler | null) => {
    if (handler) handlers.current.set(id, handler); else handlers.current.delete(id);
  }, []);
  const run = useCallback((action: () => void) => {
    if (handlers.current.size === 0) action(); else setPendingAction(() => action);
  }, []);
  const value = useMemo(() => ({ register, run }), [register, run]);
  const save = async () => {
    setSaving(true); setError(undefined);
    try {
      const snapshot = [...handlers.current.values()];
      for (const handler of snapshot) await handler.save();
      handlers.current.clear(); pendingAction?.(); setPendingAction(null);
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : '图片优化提示词保存失败');
    } finally { setSaving(false); }
  };
  return (
    <GuardContext.Provider value={value}>
      {children}
      <Modal title="文本草稿尚未保存" open={pendingAction !== null} closable={false} footer={<Space><Button onClick={() => setPendingAction(null)}>取消</Button><Button danger onClick={() => { for (const handler of handlers.current.values()) handler.discard(); handlers.current.clear(); pendingAction?.(); setPendingAction(null); }}>丢弃</Button><Button type="primary" loading={saving} onClick={() => { void save(); }}>保存</Button></Space>}>
        {error ? <Alert type="error" showIcon title={error} /> : '请选择保存、丢弃或取消。'}
      </Modal>
    </GuardContext.Provider>
  );
}

export function useUnsavedDraftGuard(): GuardValue {
  const value = useContext(GuardContext);
  return value ?? fallbackGuard;
}
