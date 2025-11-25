import type { ReactNode } from 'react';

export type Message = {
  id: string;
  role: 'user' | 'ai' | 'system';
  content: ReactNode;
  text?: string; // Add text property for TTS
  isGenerating?: boolean;
  photoDataUri?: string;
  persona?: Persona;
};

export type LocalizedString = {
  ko: string;
  en: string;
  ja: string;
  'zh-CN': string;
};

export type Persona = {
  name: LocalizedString;
  description: LocalizedString;
  avatarId: string;
  avatarUrl: string;
};

export type Language = {
  name: string;
  code: 'ko' | 'en' | 'ja' | 'zh-CN';
};
