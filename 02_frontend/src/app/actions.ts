'use server';

import {
  generateVisualExplanations,
  GenerateVisualExplanationsInput,
} from '@/ai/flows/generate-visual-explanations';
import {
  receiveSmartNearbyRecommendations,
  SmartNearbyRecommendationsInput,
} from '@/ai/flows/receive-smart-nearby-recommendations';
import { speechToText, SpeechToTextInput } from '@/ai/flows/speech-to-text';
import { textToSpeech, TextToSpeechInput } from '@/ai/flows/text-to-speech';
import { z } from 'zod';
import { revalidatePath } from 'next/cache';

// 백엔드 API 주소 (FastAPI 서버)
const API_URL = 'http://localhost:8000/api/query';

const royalAnswerSchema = z.object({
  question: z.string(),
  location: z.string().optional(),
  historicalFigurePersona: z.string().optional(),
  photoDataUri: z.string().optional(),
  language: z.string().default('ko'),
});

export async function getRoyalAnswer(formData: FormData) {
  try {
    // 1. 폼 데이터 검증
    const rawData = Object.fromEntries(formData);
    const validatedData = royalAnswerSchema.parse(rawData);

    // 2. FastAPI 백엔드로 요청 전송
    console.log('📡 백엔드로 요청 전송:', validatedData.question);
    
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        question: validatedData.question,
        location: validatedData.location || '',
        language: validatedData.language || 'ko',
        historicalFigurePersona: validatedData.historicalFigurePersona || '',
        photoDataUri: validatedData.photoDataUri || '',
      }),
      cache: 'no-store', // 항상 최신 데이터 요청
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Backend API Error: ${response.status} ${response.statusText} - ${errorText}`);
    }

    // 3. 응답 처리
    const result = await response.json();
    console.log('✅ 백엔드 응답 수신 완료');

    revalidatePath('/');
    return { success: true, answer: result.answer };

  } catch (error) {
    console.error('❌ Error in getRoyalAnswer:', error);
    const errorMessage =
      error instanceof Error ? error.message : 'An unknown error occurred.';
    return { success: false, error: errorMessage };
  }
}

const visualExplanationSchema = z.object({
  query: z.string().min(1, 'Query cannot be empty.'),
});

export async function getVisualExplanation(formData: FormData) {
  try {
    const validatedData = visualExplanationSchema.parse(
      Object.fromEntries(formData)
    );
    const result = await generateVisualExplanations(
      validatedData as GenerateVisualExplanationsInput
    );
    revalidatePath('/');
    return { success: true, videoDataUri: result.videoDataUri };
  } catch (error) {
    console.error(error);
    const errorMessage =
      error instanceof Error ? error.message : 'An unknown error occurred.';
    return { success: false, error: errorMessage };
  }
}

const recommendationsSchema = z.object({
  userLocation: z.string(),
  conversationContext: z.string(),
});

export async function getRecommendations(formData: FormData) {
  try {
    const validatedData = recommendationsSchema.parse(
      Object.fromEntries(formData)
    );
    // Hardcoded location as it's not selectable anymore
    validatedData.userLocation = 'Hanyang';
    const result = await receiveSmartNearbyRecommendations(
      validatedData as SmartNearbyRecommendationsInput
    );
    revalidatePath('/');
    return { success: true, recommendations: result.recommendations };
  } catch (error) {
    console.error(error);
    const errorMessage =
      error instanceof Error ? error.message : 'An unknown error occurred.';
    return { success: false, error: errorMessage };
  }
}

const speechToTextSchema = z.object({
  audioDataUri: z.string(),
});

export async function getTranscript(formData: FormData) {
  try {
    const validatedData = speechToTextSchema.parse(
      Object.fromEntries(formData)
    );
    const result = await speechToText(validatedData as SpeechToTextInput);
    return { success: true, text: result.text };
  } catch (error) {
    console.error(error);
    const errorMessage =
      error instanceof Error ? error.message : 'An unknown error occurred.';
    return { success: false, error: errorMessage };
  }
}

const textToSpeechSchema = z.object({
  text: z.string(),
  personaName: z.string(),
});

export async function getTextToSpeech(formData: FormData) {
  try {
    const validatedData = textToSpeechSchema.parse(
      Object.fromEntries(formData)
    );
    const result = await textToSpeech(validatedData as TextToSpeechInput);
    return { success: true, audioDataUri: result.audioDataUri };
  } catch (error) {
    console.error('TTS Error:', error);
    const errorMessage =
      error instanceof Error ? error.message : 'An unknown error occurred.';
    return { success: false, error: errorMessage };
  }
}
