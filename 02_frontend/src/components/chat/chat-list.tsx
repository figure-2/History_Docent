'use client';

import React, { useEffect, useRef } from 'react';
import type { Message, Language } from '@/lib/types';
import { ChatMessage } from './chat-message';
import { cn } from '@/lib/utils';

interface ChatListProps {
  messages: Message[];
  language: Language;
  className?: string;
}

export function ChatList({
  messages,
  language,
  className,
}: ChatListProps) {
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    listRef.current?.lastElementChild?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div
      ref={listRef}
      className={cn('container mx-auto max-w-3xl space-y-6 px-4 py-8', className)}
    >
      {messages.map((message) => (
        <ChatMessage
          key={message.id}
          message={message}
          language={language}
        />
      ))}
    </div>
  );
}
