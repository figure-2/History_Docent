'use client';

import { cn } from '@/lib/utils';
import type { Message, Language, Persona } from '@/lib/types';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { Button } from '@/components/ui/button';
import { User, Volume2, Bot, Loader2, Square } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { useState, useRef, useEffect, useCallback } from 'react';
import { getTextToSpeech } from '@/app/actions';
import { useToast } from '@/hooks/use-toast';

interface ChatMessageProps {
  message: Message;
  language: Language;
}

const translations: Record<string, Record<string, string>> = {
  ko: { title: '음성 생성 오류' },
  en: { title: 'Speech Generation Error' },
  ja: { title: '音声生成エラー' },
  'zh-CN': { title: '语音生成错误' },
};

export function ChatMessage({ message, language }: ChatMessageProps) {
  const { role, content, text, persona } = message;
  const { toast } = useToast();
  
  const [playbackState, setPlaybackState] = useState<'idle' | 'loading' | 'playing' | 'error'>('idle');
  const audioQueueRef = useRef<HTMLAudioElement[]>([]);
  const currentAudioRef = useRef<HTMLAudioElement | null>(null);
  const stopPlaybackRef = useRef<boolean>(false);

  const t = translations[language.code] || translations.ko;

  const stopPlayback = useCallback(() => {
    stopPlaybackRef.current = true;
    
    if (currentAudioRef.current) {
      currentAudioRef.current.pause();
      currentAudioRef.current.src = '';
      currentAudioRef.current = null;
    }
    
    audioQueueRef.current.forEach(audio => {
      audio.pause();
      audio.src = '';
    });
    audioQueueRef.current = [];
    
    setPlaybackState('idle');
  }, []);

  const playNextInQueue = useCallback(() => {
    if (stopPlaybackRef.current || audioQueueRef.current.length === 0) {
        if (!stopPlaybackRef.current) { // If not stopped by user, means queue is empty
            setPlaybackState('idle');
        }
      return;
    }

    setPlaybackState('playing');
    const audio = audioQueueRef.current.shift()!;
    currentAudioRef.current = audio;
      
    audio.play().catch(e => {
      console.error("Audio play failed:", e);
      stopPlayback();
    });

    audio.onended = () => {
      currentAudioRef.current = null;
      playNextInQueue();
    };
      
    audio.onerror = (e) => {
      console.error("Audio playback error:", e);
      toast({
        title: t.title,
        description: '음성 재생 중 오류가 발생했습니다.',
        variant: 'destructive',
      });
      stopPlayback();
    };
  }, [stopPlayback, t.title, toast]);

  const handlePlayAudio = async () => {
    if (playbackState === 'playing' || playbackState === 'loading') {
      stopPlayback();
      return;
    }
    
    if (!text || typeof text !== 'string' || !persona) return;

    stopPlaybackRef.current = false;
    setPlaybackState('loading');

    const sentences = text.match(/[^.!?]+[.!?\s]*/g) || [text];
    let isFirstAudio = true;

    // Process each sentence independently
    sentences.forEach(async (sentence) => {
      const trimmedSentence = sentence.trim();
      if (!trimmedSentence) return;

      try {
        const formData = new FormData();
        formData.append('text', trimmedSentence);
        formData.append('personaName', persona.name[language.code]);
        
        const result = await getTextToSpeech(formData);

        if (stopPlaybackRef.current) return;
        
        if (result.success && result.audioDataUri) {
          const newAudio = new Audio(result.audioDataUri);
          audioQueueRef.current.push(newAudio);
          
          if (isFirstAudio) {
            isFirstAudio = false;
            playNextInQueue();
          }
        } else {
          console.error('Single TTS sentence generation failed:', result.error);
          toast({
            title: t.title,
            description: result.error || '음성 변환 중 일부 문장에서 오류가 발생했습니다.',
            variant: 'destructive',
          });
        }
      } catch (error) {
        console.error('TTS request failed catastrophically:', error);
        if (!stopPlaybackRef.current) {
          toast({
              title: t.title,
              description: error instanceof Error ? error.message : 'An unexpected response was received from the server.',
              variant: 'destructive',
          });
          setPlaybackState('error');
          stopPlayback();
        }
      }
    });
  };

  useEffect(() => {
    // Cleanup on component unmount
    return stopPlayback;
  }, [stopPlayback]);
  
  useEffect(() => {
      // Reset on error state
      if (playbackState === 'error') {
          const timer = setTimeout(() => {
              setPlaybackState('idle');
          }, 2000);
          return () => clearTimeout(timer);
      }
  }, [playbackState]);

  if (role === 'system') {
    return (
      <div className="text-center text-sm text-muted-foreground italic">
        {content}
      </div>
    );
  }

  const isAi = role === 'ai';
  const personaName = isAi && persona ? persona.name[language.code] : '';

  return (
    <div
      className={cn('flex items-start gap-4', {
        'justify-end': !isAi,
      })}
    >
      {isAi && persona && (
        <Avatar className="h-10 w-10 border-2 border-primary/50">
          <AvatarImage src={persona.avatarUrl} alt={personaName} />
          <AvatarFallback>
            <Bot />
          </AvatarFallback>
        </Avatar>
      )}

      <div
        className={cn('flex max-w-[80%] flex-col gap-1', {
          'items-start': isAi,
          'items-end': !isAi,
        })}
      >
        <div className='flex items-center gap-2'>
        {isAi && (
          <span className="text-sm font-semibold text-foreground">
            {personaName}
          </span>
        )}
        {isAi && text && (
            <Button
              size="icon"
              variant="ghost"
              className="h-6 w-6"
              onClick={handlePlayAudio}
              disabled={playbackState === 'error'}
            >
              {(playbackState === 'loading') && <Loader2 className="animate-spin" />}
              {(playbackState === 'playing') && <Square className="h-4 w-4 fill-current" />}
              {(playbackState === 'idle' || playbackState === 'error') && <Volume2 className="text-muted-foreground" />}
            </Button>
          )}
        </div>
        <Card
          className={cn('rounded-2xl p-0 overflow-hidden', {
            'bg-card': isAi,
            'bg-primary text-primary-foreground': !isAi,
          })}
        >
          <CardContent className="p-3 text-sm">{content}</CardContent>
        </Card>
      </div>

      {!isAi && (
        <Avatar className="h-10 w-10 border bg-secondary">
          <AvatarFallback>
            <User className="h-5 w-5" />
          </AvatarFallback>
        </Avatar>
      )}
    </div>
  );
}
