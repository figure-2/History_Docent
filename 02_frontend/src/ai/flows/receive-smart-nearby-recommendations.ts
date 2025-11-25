'use server';
/**
 * @fileOverview Provides recommendations for nearby attractions and dining options based on the user's current location and the conversation context.
 *
 * - receiveSmartNearbyRecommendations - A function that retrieves smart nearby recommendations.
 * - SmartNearbyRecommendationsInput - The input type for the receiveSmartNearbyRecommendations function.
 * - SmartNearbyRecommendationsOutput - The return type for the receiveSmartNearbyRecommendations function.
 */

import {ai} from '@/ai/genkit';
import {z} from 'genkit';

const SmartNearbyRecommendationsInputSchema = z.object({
  userLocation: z
    .string()
    .describe('The current GPS coordinates of the user.'),
  conversationContext: z
    .string()
    .describe('The ongoing conversation with the AI docent.'),
});
export type SmartNearbyRecommendationsInput = z.infer<
  typeof SmartNearbyRecommendationsInputSchema
>;

const SmartNearbyRecommendationsOutputSchema = z.object({
  recommendations: z
    .string()
    .describe(
      'A list of nearby attractions and dining options based on the user location and conversation context.'
    ),
});
export type SmartNearbyRecommendationsOutput = z.infer<
  typeof SmartNearbyRecommendationsOutputSchema
>;

export async function receiveSmartNearbyRecommendations(
  input: SmartNearbyRecommendationsInput
): Promise<SmartNearbyRecommendationsOutput> {
  return receiveSmartNearbyRecommendationsFlow(input);
}

const prompt = ai.definePrompt({
  name: 'smartNearbyRecommendationsPrompt',
  input: {schema: SmartNearbyRecommendationsInputSchema},
  output: {schema: SmartNearbyRecommendationsOutputSchema},
  prompt: `Based on the user's location ({{{userLocation}}}) and the current conversation about Hanyang ({{{conversationContext}}}), provide recommendations for nearby attractions and dining options. Return the output in a single paragraph.`,
});

const receiveSmartNearbyRecommendationsFlow = ai.defineFlow(
  {
    name: 'receiveSmartNearbyRecommendationsFlow',
    inputSchema: SmartNearbyRecommendationsInputSchema,
    outputSchema: SmartNearbyRecommendationsOutputSchema,
  },
  async input => {
    const {output} = await prompt(input);
    return output!;
  }
);
