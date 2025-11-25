'use client';

import { Paintbrush } from 'lucide-react';
import { cn } from '@/lib/utils';

export const LoadingIndicator = ({
  className,
  text,
}: {
  className?: string;
  text?: string;
}) => {
  return (
    <div
      className={cn('flex items-center justify-center gap-2', className)}
    >
      <Paintbrush className="h-5 w-5 animate-spin text-primary" />
      {text && <span className="text-sm text-muted-foreground">{text}</span>}
    </div>
  );
};
