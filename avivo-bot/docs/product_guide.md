# Product Guide

## What is AvivaBot?

AvivaBot is an AI-powered enterprise assistant built on a hybrid pipeline combining Retrieval-Augmented
Generation (RAG) and Computer Vision. It answers questions from your company knowledge base,
describes images with AI-generated captions, and supports interactive sessions via Telegram and a web UI.

## Key Features

**1. Ask Anything (RAG Mode)**
Ask any question in natural language. AvivaBot retrieves relevant context from your company knowledge
base (policies, FAQs, product docs) and generates an accurate, grounded answer using a local LLM.
No hallucinations — every answer cites the source document.

**2. Describe Image (Vision Mode)**
Upload any image and AvivaBot will return a 2-sentence description plus 3 keyword tags.
Powered by LLaVA, a state-of-the-art open-source multimodal model running entirely locally.

**3. Session Memory**
AvivaBot remembers the last 3 turns of your conversation, enabling follow-up questions without
repeating context. Session memory expires after 24 hours automatically.

**4. Web Frontend**
Access AvivaBot directly from your browser at the public Gradio share URL — no Telegram required.
The web UI offers the same Ask and Vision capabilities with a beautiful, responsive interface.

## Pricing

| Plan       | Messages/Day | Image Analyses | History Turns | Price         |
|------------|-------------|----------------|---------------|---------------|
| Free       | 20          | 5              | 1 turn        | ₹0/month      |
| Pro        | 200         | 50             | 3 turns       | ₹999/month    |
| Enterprise | Unlimited   | Unlimited      | 5 turns       | Custom pricing|

## How Answers Are Generated

1. Your query is embedded into a high-dimensional vector using `all-MiniLM-L6-v2`.
2. The top-3 most semantically similar chunks are retrieved from ChromaDB.
3. A carefully engineered prompt is built with system instructions, your chat history, retrieved context,
   and your question.
4. The prompt is sent to `phi3:mini` (a 3.8B-parameter model) running on Ollama.
5. The answer is returned with source attribution.

## Supported Document Types for Knowledge Base

- Markdown (.md) — primary format, best supported
- Plain text (.txt)
- PDF (coming soon)
- Confluence pages (via API integration, Enterprise plan)

## Limitations

- AvivaBot only knows what is in its knowledge base. It cannot browse the internet.
- Vision analysis works best with clear, high-resolution images (≥512×512 pixels).
- Responses may take 5–15 seconds depending on server load and model size.
- The Ollama LLMs run locally; no data is sent to external APIs.

## Roadmap

Q2 2025: PDF ingestion support, voice input via Whisper, multi-language support.
Q3 2025: Confluence and Notion connectors, admin dashboard, usage analytics.
Q4 2025: Fine-tuning on company-specific data, custom model hosting.
