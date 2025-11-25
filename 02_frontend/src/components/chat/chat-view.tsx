'use client';

import React, {
  useState,
  useOptimistic,
  useRef,
  useEffect,
  useTransition,
} from 'react';
import Image from 'next/image';
import { AppHeader } from '@/components/layout/header';
import { ChatList } from '@/components/chat/chat-list';
import { ChatInputForm } from '@/components/chat/chat-input-form';
import { HISTORICAL_PERSONAS, LANGUAGES } from '@/lib/constants';
import type { Message, Language, Persona } from '@/lib/types';
import {
  getRoyalAnswer,
  getVisualExplanation,
  getRecommendations,
  getTranscript,
} from '@/app/actions';
import { useToast } from '@/hooks/use-toast';
import { LoadingIndicator } from '../ui/loading-indicator';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '../ui/card';

// 브라우저 호환 UUID 생성 함수
function generateUUID(): string {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  // Fallback: crypto.getRandomValues를 사용한 UUID v4 생성
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    const r = (Math.random() * 16) | 0;
    const v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

const translations: Record<string, Record<string, string>> = {
  ko: {
    welcome:
      '안녕하세요! 저는 여러분의 AI 도슨트입니다. 한양의 역사와 건축, 문화에 대해 무엇이든 물어보세요.',
    recommendationSystemMessage: '주변 추천 장소입니다.',
    generatingRecommendations: '주변 추천 장소를 찾고 있습니다...',
    noQueryForVisual: '시각화할 쿼리 없음',
    noQueryDescription: '동영상을 생성하기 전에 먼저 질문을 해주세요.',
    generatingVisual: '영상 설명을 생성 중입니다... 잠시만 기다려 주세요.',
    visualFor: '다음에 대한 시각적 설명 생성 중: ',
    imageAttached: '이미지 첨부됨',
    imageCaption:
      '이 이미지에 대해 질문하거나 AI가 설명하도록 할 수 있습니다.',
    recording: '녹음 중...',
    transcribing: '음성을 텍스트로 변환 중...',
    imageDescriptionPrompt:
      '이 이미지에 대해 설명하고, 이미지에 대한 흥미로운 사실이나 관련된 역사적 맥락을 알려주세요.',
    recordingError: '녹음 오류',
    recordingErrorDescription:
      '녹음을 시작할 수 없습니다. 마이크 권한을 확인해주세요.',
  },
  en: {
    welcome:
      'Hello! I am your AI docent. Ask me anything about the history, architecture, and culture of Hanyang.',
    recommendationSystemMessage: 'Here are some nearby recommendations.',
    generatingRecommendations: 'Finding nearby recommendations...',
    noQueryForVisual: 'No Query to Visualize',
    noQueryDescription: 'Please ask a question first before generating a video.',
    generatingVisual: 'Generating video explanation... please wait.',
    visualFor: 'Generating visual explanation for: ',
    imageAttached: 'Image Attached',
    imageCaption:
      'You can ask a question about this image or let the AI describe it.',
    recording: 'Recording...',
    transcribing: 'Transcribing audio...',
    imageDescriptionPrompt:
      'Please describe this image, and tell me any interesting facts or historical context related to it.',
    recordingError: 'Recording Error',
    recordingErrorDescription:
      'Could not start recording. Please check microphone permissions.',
  },
  ja: {
    welcome:
      'こんにちは！私はあなたのAIドーセントです。漢陽の歴史、建築、文化について何でもお尋ねください。',
    recommendationSystemMessage: '近くのおすすめスポットです。',
    generatingRecommendations: '近くのおすすめスポットを検索しています...',
    noQueryForVisual: '視覚化するクエリがありません',
    noQueryDescription: '動画を生成する前に、まず質問をしてください。',
    generatingVisual: '映像説明を生成中です...しばらくお待ちください。',
    visualFor: '次の視覚的説明を生成しています: ',
    imageAttached: '画像が添付されました',
    imageCaption: 'この画像について質問したり、AIに説明させたりすることができます。',
    recording: '録音中...',
    transcribing: '音声をテキストに変換しています...',
    imageDescriptionPrompt:
      'この画像を説明し、関連する興味深い事実や歴史的背景を教えてください。',
    recordingError: '録音エラー',
    recordingErrorDescription:
      '録音を開始できませんでした。マイクの許可を確認してください。',
  },
  'zh-CN': {
    welcome:
      '您好！我是您的AI导览员。关于汉阳的历史、建筑和文化，您可以问我任何问题。',
    recommendationSystemMessage: '这是附近的推荐。',
    generatingRecommendations: '正在寻找附近的推荐...',
    noQueryForVisual: '没有可供可视化的查询',
    noQueryDescription: '在生成视频之前，请先提问。',
    generatingVisual: '正在生成视频讲解...请稍候。',
    imageAttached: '图片已附加',
    imageCaption: '您可以就此图片提问或让AI进行描述。',
    recording: '录音中...',
    transcribing: '正在转录音频...',
    imageDescriptionPrompt:
      '请描述这张图片，并告诉我任何与之相关的有趣事实或历史背景。',
    recordingError: '录音错误',
    recordingErrorDescription: '无法开始录音。请检查麦克风权限。',
  },
};

export function ChatView() {
  const { toast } = useToast();
  const [language, setLanguage] = useState<Language>(LANGUAGES[0]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [persona, setPersona] = useState<Persona>(HISTORICAL_PERSONAS[0]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isPending, startTransition] = useTransition();

  const [optimisticMessages, addOptimisticMessage] = useOptimistic<
    Message[],
    Message | Message[]
  >(messages, (state, newMessages) => [
    ...state,
    ...(Array.isArray(newMessages) ? newMessages : [newMessages]),
  ]);

  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  useEffect(() => {
    const welcomeMessage: Message = {
      id: 'init',
      role: 'system' as const,
      content: translations[language.code].welcome,
      persona: HISTORICAL_PERSONAS.find(p => p.avatarId === 'history-guide') || HISTORICAL_PERSONAS[5],
    };
    setMessages([welcomeMessage]);
  }, [language]);

  const handleLanguageChange = (lang: typeof LANGUAGES[0]) => {
    setLanguage(lang);
  };

  const handleSendMessage = async (
    messageContent: string,
    photoDataUri?: string
  ) => {
    if (!messageContent && !photoDataUri) return;

    const currentPersona = persona;

    const userMessage: Message = {
      id: generateUUID(),
      role: 'user',
      content: messageContent,
      text: messageContent,
      photoDataUri: photoDataUri,
    };

    const aiResponsePlaceholder: Message = {
      id: generateUUID(),
      role: 'ai',
      content: <LoadingIndicator />,
      isGenerating: true,
      persona: currentPersona,
    };
    
    startTransition(() => {
      addOptimisticMessage([userMessage, aiResponsePlaceholder]);
    });
    
    setIsGenerating(true);

    const formData = new FormData();
    formData.append('question', messageContent);
    formData.append('location', 'Hanyang');
    formData.append('historicalFigurePersona', currentPersona.name[language.code]);
    if (photoDataUri) {
      formData.append('photoDataUri', photoDataUri);
    }
    formData.append('language', language.code);

    const result = await getRoyalAnswer(formData);

    const aiResponseMessage: Message = {
      id: generateUUID(), // 새로운 UUID 생성
      role: 'ai',
      content: result.success
        ? result.answer
        : `Error: ${result.error || 'Could not fetch answer.'}`,
      text: result.success ? result.answer : undefined,
      isGenerating: false,
      persona: currentPersona,
    };
    
    setMessages(prev => [...prev.filter(m => m.id !== userMessage.id && m.id !== aiResponsePlaceholder.id), userMessage, aiResponseMessage]);
    setIsGenerating(false);
  };

  const handleImageSelected = (photoDataUri: string) => {
    const t = translations[language.code];
    const imageMessageContent = t.imageDescriptionPrompt;
    
    const currentPersona = persona;

    const userMessage: Message = {
      id: generateUUID(),
      role: 'user',
      content: (
        <Card className="w-64">
          <CardHeader>
            <CardTitle>{t.imageAttached}</CardTitle>
            <CardDescription>{t.imageCaption}</CardDescription>
          </CardHeader>
          <CardContent>
            <Image
              src={photoDataUri}
              alt="Uploaded image"
              width={200}
              height={200}
              className="rounded-md object-cover"
            />
          </CardContent>
        </Card>
      ),
      text: imageMessageContent,
      photoDataUri: photoDataUri,
    };

    const aiResponsePlaceholder: Message = {
      id: generateUUID(),
      role: 'ai',
      content: <LoadingIndicator />,
      isGenerating: true,
      persona: currentPersona,
    };
    
    startTransition(() => {
      addOptimisticMessage([userMessage, aiResponsePlaceholder]);
    });
    setIsGenerating(true);

    const formData = new FormData();
    formData.append('question', imageMessageContent);
    formData.append('location', 'Hanyang');
    formData.append('historicalFigurePersona', currentPersona.name[language.code]);
    formData.append('photoDataUri', photoDataUri);
    formData.append('language', language.code);

    getRoyalAnswer(formData).then(result => {
      const aiResponseMessage: Message = {
        id: generateUUID(), // 새로운 UUID 생성
        role: 'ai',
        content: result.success
          ? result.answer
          : `Error: ${result.error || 'Could not fetch answer.'}`,
        text: result.success ? result.answer : undefined,
        isGenerating: false,
        persona: currentPersona,
      };
      setMessages(prev => [...prev.filter(m => m.id !== userMessage.id && m.id !== aiResponsePlaceholder.id), userMessage, aiResponseMessage]);
      setIsGenerating(false);
    });
  };

  const handleToggleRecording = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  const startRecording = async () => {
    const t = translations[language.code];
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      setIsRecording(true);

      const systemMessage: Message = {
        id: generateUUID(),
        role: 'system',
        content: t.recording,
      };
      startTransition(() => {
        addOptimisticMessage(systemMessage);
      });

      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = event => {
        audioChunksRef.current.push(event.data);
      };

      mediaRecorderRef.current.onstop = async () => {
        setMessages(prev => prev.filter(m => m.id !== systemMessage.id));

        const transcribingMessage: Message = {
          id: generateUUID(),
          role: 'system',
          content: t.transcribing,
        };

        startTransition(() => {
          addOptimisticMessage(transcribingMessage);
        });

        const audioBlob = new Blob(audioChunksRef.current, {
          type: 'audio/webm',
        });
        const reader = new FileReader();
        reader.readAsDataURL(audioBlob);
        reader.onloadend = async () => {
          const base64Audio = reader.result as string;

          const formData = new FormData();
          formData.append('audioDataUri', base64Audio);
          const result = await getTranscript(formData);

          setMessages(prev => prev.filter(m => m.id !== transcribingMessage.id));

          if (result.success && result.text) {
            handleSendMessage(result.text);
          } else {
            toast({
              title: 'Error transcribing audio',
              description: result.error || 'Could not transcribe audio.',
              variant: 'destructive',
            });
          }
        };
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorderRef.current.start();
    } catch (error) {
      console.error('Error starting recording:', error);
      setIsRecording(false);
      toast({
        title: t.recordingError,
        description: t.recordingErrorDescription,
        variant: 'destructive',
      });
    }
  };

  const stopRecording = () => {
    if (
      mediaRecorderRef.current &&
      mediaRecorderRef.current.state === 'recording'
    ) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  return (
    <div className="flex h-screen flex-col bg-background">
      <AppHeader
        language={language}
        onLanguageChange={handleLanguageChange}
        persona={persona}
        onPersonaChange={setPersona}
      />
      <main className="flex-1 overflow-y-auto">
        <ChatList
          messages={optimisticMessages}
          language={language}
        />
      </main>
      <footer className="sticky bottom-0 border-t bg-background/95 p-4">
        <ChatInputForm
          onSendMessage={handleSendMessage}
          onImageSelected={handleImageSelected}
          isGenerating={isGenerating || isPending}
          language={language.code}
          isRecording={isRecording}
          onToggleRecording={handleToggleRecording}
        />
      </footer>
    </div>
  );
}
