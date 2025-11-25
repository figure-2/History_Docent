# **App Name**: GungAI: AI Royal Docent

## Core Features:

- Location-Aware 'Royal Answers': Answer user questions about Gyeongbokgung based on its location within the palace using a persona of historical figures. The AI uses the RAG pipeline to use reliable and vast historical resources.
- History Comes Alive: Generate short videos using the Veo API to answer questions that need visual explanations.
- Smart Nearby Recommendations: Provide recommendations for nearby attractions, dining options, based on the user's location and conversation context.
- Immersive Chat UI: The main UI displays a chat window with messages in speech bubbles.
- RAG Pipeline: Uses Langchain to orchestrate data retrieval from Vertex AI Vector Search and inject persona prompts into the LLM.
- Firebase Integration: Uses Firebase for hosting, authentication, storage, and Firestore for storing chat history, user profiles, and recommendation data.

## Style Guidelines:

- Primary color: #FFD700 (Gold) - Symbolizes royal authority and history.
- Background color: #F5F5DC (Beige) - Creates a soft, antique background.
- Accent color: #DC143C (Crimson) - Used for primary buttons or notifications to draw attention.
- Headline font: 'Nanum Myeongjo' (Serif) - Used for titles and the app name to create an elegant, historical feel.
- Body font: 'Noto Sans KR' (Sans-serif) - Used for all body text to ensure maximum readability.
- Uses icons motifed from traditional Korean patterns and royal symbols.
- Intuitive layout with a fixed header and footer, and a scrollable chat body.
- Subtle loading animation (e.g., a brushstroke effect) is displayed while the AI generates a response to maintain immersion.