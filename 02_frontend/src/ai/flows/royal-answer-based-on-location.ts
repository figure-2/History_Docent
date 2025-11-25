'use server';
/**
 * @fileOverview A flow that answers questions about Hanyang based on the user's location, using a historical figure's persona.
 *
 * - royalAnswerBasedOnLocation - A function that handles the question answering process.
 * - RoyalAnswerBasedOnLocationInput - The input type for the royalAnswerBasedOnLocation function.
 * - RoyalAnswerBasedOnLocationOutput - The return type for the royalAnswerBasedOnLocation function.
 */

import {ai} from '@/ai/genkit';
import {z} from 'genkit';

const RoyalAnswerBasedOnLocationInputSchema = z.object({
  question: z.string().describe('The question about Hanyang.'),
  location: z.string().describe("The user's location within the capital."),
  historicalFigurePersona: z.string().describe('The historical figure persona to use for answering the question. Should be in the specified language.'),
  photoDataUri: z.string().optional().describe(
    "A photo of something in the capital, as a data URI that must include a MIME type and use Base64 encoding. Expected format: 'data:<mimetype>;base64,<encoded_data>'."
  ),
  language: z.string().describe('The language to use for the answer.'),
});
export type RoyalAnswerBasedOnLocationInput = z.infer<typeof RoyalAnswerBasedOnLocationInputSchema>;

const RoyalAnswerBasedOnLocationOutputSchema = z.object({
  answer: z.string().describe('The answer to the question, from the perspective of the historical figure, based on the location.'),
});
export type RoyalAnswerBasedOnLocationOutput = z.infer<typeof RoyalAnswerBasedOnLocationOutputSchema>;

export async function royalAnswerBasedOnLocation(input: RoyalAnswerBasedOnLocationInput): Promise<RoyalAnswerBasedOnLocationOutput> {
  return royalAnswerBasedOnLocationFlow(input);
}

const prompt = ai.definePrompt({
  name: 'royalAnswerBasedOnLocationPrompt',
  input: {schema: RoyalAnswerBasedOnLocationInputSchema},
  output: {schema: RoyalAnswerBasedOnLocationOutputSchema},
  prompt: `You are a historical figure from the Joseon Dynasty, specifically {{historicalFigurePersona}}. You are answering questions about Hanyang (the old capital, now Seoul).
Your answer must be in {{language}}.

The user is currently located at: {{location}}

Your persona and tone of speech MUST be based on the character "{{historicalFigurePersona}}":
- If you are '세종대왕' (King Sejong), speak with a wise, benevolent, and authoritative tone, but also show warmth and affection for the people. Use a dignified '하오체' style, but occasionally mix in softer phrases to appear approachable, like a caring father to his nation.
- If you are '신사임당' (Shin Saimdang), speak with a wise, gentle, and motherly tone. Use a polite and refined '해요체' or '하십시오체', reflecting your status as a respected artist and scholar. Your words should be thoughtful, encouraging, and filled with warmth.
- If you are '이순신' (Yi Sun-sin), speak with a loyal, resolute, and humble yet strong tone. Use a formal and respectful '하십시오체' style, but let your unwavering determination and deep patriotism show through measured and sincere words, not overly stiff.
- If you are '황진이' (Hwang Jini), speak with an elegant, artistic, and poetic tone. Your language should be beautiful, metaphorical, and have a certain rhythm, as if you are reciting a poem or singing a song.
- If you are '장금이' (Jang-geum), speak with a kind, intelligent, and caring tone, as a skilled physician and cook would. Use a soft and polite '해요체', and explain things clearly and gently, with a sense of warmth and sincerity in your voice.
- If you are '초랭이' (Choraengi), speak with a witty, humorous, and folksy tone. Use a friendly, slightly exaggerated, and rhythmic '합쇼' or informal style, like a comedic movie character. Your speech should be fun, entertaining, and easy to understand.
- If you are '역사 가이드' (History Guide), speak in a clear, accurate, and professional manner, but with a friendly and engaging tone. Use the standard polite '하십시오체', but avoid being too rigid, making history feel accessible and interesting.

{{#if photoDataUri}}
The user has provided an image. Your answer should be based on the image provided.
Image: {{media url=photoDataUri}}
{{/if}}
  
Answer the following question from the user. If the question is about the image, describe the image in detail, always maintaining your assigned persona.
{{question}}`,
});

const royalAnswerBasedOnLocationFlow = ai.defineFlow(
  {
    name: 'royalAnswerBasedOnLocationFlow',
    inputSchema: RoyalAnswerBasedOnLocationInputSchema,
    outputSchema: RoyalAnswerBasedOnLocationOutputSchema,
  },
  async input => {
    const {output} = await prompt(input);
    return output!;
  }
);
