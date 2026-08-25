export function subscribeBeforeUnload(listener: (event: BeforeUnloadEvent) => void): () => void {
  self.addEventListener('beforeunload', listener as EventListener);
  return () => self.removeEventListener('beforeunload', listener as EventListener);
}
