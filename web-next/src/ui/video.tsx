import type { ComponentPropsWithoutRef, ReactNode } from 'react';

export interface VideoProps extends Omit<ComponentPropsWithoutRef<'video'>, 'aria-label' | 'children' | 'style'> {
  children?: ReactNode;
  label: string;
}

export function Video({ children, label, ...props }: VideoProps) {
  return (
    <video aria-label={label} {...props}>
      {children}
    </video>
  );
}
