'use server';

/**
 * @fileOverview Generates short videos using the Veo API to answer questions that need visual explanations.
 *
 * - generateVisualExplanations - A function that generates a video explanation.
 * - GenerateVisualExplanationsInput - The input type for the generateVisualExplanations function.
 * - GenerateVisualExplanationsOutput - The return type for the generateVisualExplanations function.
 */

import {ai} from '@/ai/genkit';
import {z} from 'genkit';
import { googleAI } from '@genkit-ai/google-genai';
import * as fs from 'fs';
import { Readable } from 'stream';
import { MediaPart } from 'genkit';

const GenerateVisualExplanationsInputSchema = z.object({
  query: z.string().describe('The query to generate a visual explanation for.'),
});

export type GenerateVisualExplanationsInput = z.infer<typeof GenerateVisualExplanationsInputSchema>;

const GenerateVisualExplanationsOutputSchema = z.object({
  videoDataUri: z.string().describe('The generated video as a data URI.'),
});

export type GenerateVisualExplanationsOutput = z.infer<typeof GenerateVisualExplanationsOutputSchema>;

export async function generateVisualExplanations(input: GenerateVisualExplanationsInput): Promise<GenerateVisualExplanationsOutput> {
  return generateVisualExplanationsFlow(input);
}

const generateVisualExplanationsFlow = ai.defineFlow(
  {
    name: 'generateVisualExplanationsFlow',
    inputSchema: GenerateVisualExplanationsInputSchema,
    outputSchema: GenerateVisualExplanationsOutputSchema,
  },
  async (input) => {
    let { operation } = await ai.generate({
      model: googleAI.model('veo-2.0-generate-001'),
      prompt: input.query,
      config: {
        durationSeconds: 5,
        aspectRatio: '16:9',
      },
    });

    if (!operation) {
      throw new Error('Expected the model to return an operation');
    }

    // Wait until the operation completes. Note that this may take some time, maybe even up to a minute. Design the UI accordingly.
    while (!operation.done) {
      operation = await ai.checkOperation(operation);
      // Sleep for 5 seconds before checking again.
      await new Promise((resolve) => setTimeout(resolve, 5000));
    }

    if (operation.error) {
      throw new Error('failed to generate video: ' + operation.error.message);
    }

    const video = operation.output?.message?.content.find((p) => !!p.media);
    if (!video) {
      throw new Error('Failed to find the generated video');
    }
    const videoDataUri = await downloadVideo(video);
    return { videoDataUri };
  }
);

async function downloadVideo(video: MediaPart): Promise<string> {
  const fetch = (await import('node-fetch')).default;
  // Add API key before fetching the video.
  const videoDownloadResponse = await fetch(
    `${video.media!.url}&key=${process.env.GEMINI_API_KEY}`
  );
  if (
    !videoDownloadResponse ||
    videoDownloadResponse.status !== 200 ||
    !videoDownloadResponse.body
  ) {
    throw new Error('Failed to fetch video');
  }
  const buffer = await videoDownloadResponse.arrayBuffer();
  const base64 = Buffer.from(buffer).toString('base64');

  return `data:video/mp4;base64,${base64}`;
}

