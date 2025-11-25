'use server';

/**
 * @fileOverview Converts text to speech using a persona-based voice.
 *
 * - textToSpeech - A function that converts text to speech.
 * - TextToSpeechInput - The input type for the textToSpeech function.
 * - TextToSpeechOutput - The return type for the textToSpeech function.
 */

import { ai } from '@/ai/genkit';
import { z } from 'zod';
import { googleAI } from '@genkit-ai/google-genai';
import wav from 'wav';

// Define voice mapping based on persona
const voiceMap: Record<string, string> = {
  // 세종대왕: 위엄과 따뜻함을 겸비한 톤
  '세종대왕': 'algenib', // More authoritative male voice
  'King Sejong': 'algenib',
  '世宗大王': 'algenib',
  
  // 신사임당: 지혜롭고 온화한 어머니의 톤
  '신사임당': 'autonoe', 
  'Shin Saimdang': 'autonoe',
  '申師任堂': 'autonoe',
  '申师任堂': 'autonoe',

  // 이순신: 강직하고 진지하며 담백한 톤
  '이순신': 'sadachbia', // Resolute and strong male voice
  'Yi Sun-sin': 'sadachbia',
  '李舜臣': 'sadachbia',

  // 황진이: 우아하고 감성적이며 세련된 톤
  '황진이': 'callirrhoe',
  'Hwang Jini': 'callirrhoe',
  '黄真伊': 'callirrhoe',

  // 장금이: 따뜻하고 부드러우며 섬세한 톤
  '장금이': 'autonoe',
  'Jang-geum': 'autonoe',
  'チャングム': 'autonoe',
  '长今': 'autonoe',
  
  // 초랭이: 재치있고 민속적이며 경쾌한 톤
  '초랭이': 'puck',
  'Choraengi': 'puck',
  'チョレンイ': 'puck',
  '草랭이': 'puck',
  
  // 역사 가이드: 명확하고 친절한 톤
  '역사 가이드': 'vindemiatrix', // Clear and friendly guide voice
  'History Guide': 'vindemiatrix',
  '歴史ガイド': 'vindemiatrix',
  '历史向导': 'vindemiatrix',
};

const TextToSpeechInputSchema = z.object({
  text: z.string().describe('The text to be converted to speech.'),
  personaName: z.string().describe('The name of the persona to determine the voice.'),
});
export type TextToSpeechInput = z.infer<typeof TextToSpeechInputSchema>;

const TextToSpeechOutputSchema = z.object({
  audioDataUri: z.string().describe('The generated audio as a data URI.'),
});
export type TextToSpeechOutput = z.infer<typeof TextToSpeechOutputSchema>;

export async function textToSpeech(
  input: TextToSpeechInput
): Promise<TextToSpeechOutput> {
  return textToSpeechFlow(input);
}

const textToSpeechFlow = ai.defineFlow(
  {
    name: 'textToSpeechFlow',
    inputSchema: TextToSpeechInputSchema,
    outputSchema: TextToSpeechOutputSchema,
  },
  async ({ text, personaName }) => {
    const voiceName = voiceMap[personaName] || 'puck'; // Default to puck if no match

    const { media } = await ai.generate({
      model: googleAI.model('gemini-2.5-flash-preview-tts'),
      config: {
        responseModalities: ['AUDIO'],
        speechConfig: {
          voiceConfig: {
            prebuiltVoiceConfig: { voiceName },
          },
        },
      },
      prompt: text,
    });

    if (!media) {
      throw new Error('No audio data was returned from the TTS model.');
    }

    const audioBuffer = Buffer.from(
      media.url.substring(media.url.indexOf(',') + 1),
      'base64'
    );
    
    const wavBase64 = await toWav(audioBuffer);
    const audioDataUri = `data:audio/wav;base64,${wavBase64}`;

    return { audioDataUri };
  }
);

async function toWav(
  pcmData: Buffer,
  channels = 1,
  rate = 24000,
  sampleWidth = 2
): Promise<string> {
  return new Promise((resolve, reject) => {
    const writer = new wav.Writer({
      channels,
      sampleRate: rate,
      bitDepth: sampleWidth * 8,
    });

    const bufs: any[] = [];
    writer.on('error', reject);
    writer.on('data', (d) => {
      bufs.push(d);
    });
    writer.on('end', () => {
      resolve(Buffer.concat(bufs).toString('base64'));
    });

    writer.write(pcmData);
    writer.end();
  });
}
