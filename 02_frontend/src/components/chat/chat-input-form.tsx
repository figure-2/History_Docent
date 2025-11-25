'use client';

import { zodResolver } from '@hookform/resolvers/zod';
import { useForm } from 'react-hook-form';
import { z } from 'zod';
import { Button } from '@/components/ui/button';
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormMessage,
} from '@/components/ui/form';
import { Textarea } from '@/components/ui/textarea';
import { Send, ImagePlus, Mic, MicOff } from 'lucide-react';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '../ui/tooltip';
import { useRef } from 'react';

const translations: Record<string, Record<string, string>> = {
  ko: {
    messageCannotBeEmpty: '메시지는 비워둘 수 없습니다.',
    placeholderFollowUp: '후속 질문을 하시거나 새로운 것을 물어보세요~',
    placeholderDefault: '궁금한 것을 물어보세요~',
    uploadImage: '이미지 업로드',
    startRecording: '음성 녹음 시작',
    stopRecording: '음성 녹음 중지',
    sendMessage: '메시지 보내기',
  },
  en: {
    messageCannotBeEmpty: 'Message cannot be empty.',
    placeholderFollowUp:
      'Ask a follow-up question or ask something new about the palace...',
    placeholderDefault: 'Ask a question about the palace...',
    uploadImage: 'Upload image',
    startRecording: 'Start recording',
    stopRecording: 'Stop recording',
    sendMessage: 'Send message',
  },
  ja: {
    messageCannotBeEmpty: 'メッセージを空にすることはできません。',
    placeholderFollowUp:
      '宮殿について追加の質問をするか、新しい質問をしてください...',
    placeholderDefault: '宮殿について質問してください...',
    uploadImage: '画像をアップロード',
    startRecording: '録音を開始',
    stopRecording: '録音を停止',
    sendMessage: 'メッセージを送信',
  },
  'zh-CN': {
    messageCannotBeEmpty: '消息不能为空。',
    placeholderFollowUp: '提出后续问题或询问有关宫殿的新问题...',
    placeholderDefault: '问一个关于宫殿的问题...',
    uploadImage: '上传图片',
    startRecording: '开始录音',
    stopRecording: '停止录音',
    sendMessage: '发送消息',
  },
};

interface ChatInputFormProps {
  onSendMessage: (message: string, photoDataUri?: string) => void;
  onImageSelected: (photoDataUri: string) => void;
  isGenerating: boolean;
  language: string;
  isRecording: boolean;
  onToggleRecording: () => void;
}

export function ChatInputForm({
  onSendMessage,
  onImageSelected,
  isGenerating,
  language,
  isRecording,
  onToggleRecording,
}: ChatInputFormProps) {
  const t = translations[language] || translations.ko;
  const fileInputRef = useRef<HTMLInputElement>(null);

  const formSchema = z.object({
    message: z.string().min(1, t.messageCannotBeEmpty),
  });

  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      message: '',
    },
  });

  function onSubmit(values: z.infer<typeof formSchema>) {
    onSendMessage(values.message);
    form.reset();
  }

  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      if (!isGenerating && form.getValues('message').trim()) {
        form.handleSubmit(onSubmit)();
      }
    }
  };

  const handleImageFileChange = (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = e => {
        const photoDataUri = e.target?.result as string;
        onImageSelected(photoDataUri);
      };
      reader.readAsDataURL(file);
    }
  };

  return (
    <TooltipProvider>
      <Form {...form}>
        <form
          onSubmit={form.handleSubmit(onSubmit)}
          className="flex items-start gap-4"
        >
          <div className="flex items-center gap-1">
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleImageFileChange}
              className="hidden"
              accept="image/*"
            />
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  type="button"
                  size="icon"
                  variant="ghost"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isGenerating}
                >
                  <ImagePlus className="h-5 w-5 text-primary" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                <p>{t.uploadImage}</p>
              </TooltipContent>
            </Tooltip>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  type="button"
                  size="icon"
                  variant={isRecording ? 'destructive' : 'ghost'}
                  onClick={onToggleRecording}
                  disabled={isGenerating}
                >
                  {isRecording ? (
                    <MicOff className="h-5 w-5" />
                  ) : (
                    <Mic className="h-5 w-5 text-primary" />
                  )}
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                <p>{isRecording ? t.stopRecording : t.startRecording}</p>
              </TooltipContent>
            </Tooltip>
          </div>

          <div className="flex-1 relative">
            <FormField
              control={form.control}
              name="message"
              render={({ field }) => (
                <FormItem>
                  <FormControl>
                    <Textarea
                      placeholder={
                        form.watch('message')
                          ? t.placeholderFollowUp
                          : t.placeholderDefault
                      }
                      className="min-h-12 resize-none"
                      {...field}
                      onKeyDown={handleKeyDown}
                      disabled={isGenerating}
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
          </div>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button type="submit" size="icon" disabled={isGenerating}>
                <Send className="h-5 w-5" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              <p>{t.sendMessage}</p>
            </TooltipContent>
          </Tooltip>
        </form>
      </Form>
    </TooltipProvider>
  );
}
