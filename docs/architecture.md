# Architecture

```mermaid
flowchart LR
    A[Local MP3/WAV/audio file] --> B[Validate extension and 25 MB limit]
    B --> C[HF ASR openai/whisper-large-v3]
    C --> D[Transcript text]
    D --> E[HF Featherless Qwen chat model]
    E --> F[JSON object mode]
    F --> G[Pydantic validation]
    G --> H[JSON or Markdown report]
```

## Responsibilities

- **Local Python:** validates the file, opens the upload, orchestrates the two API calls, validates the result, and renders output.
- **Whisper:** converts speech to a transcript. It does not produce action items.
- **Qwen or Llama 3.3:** reads the transcript and extracts intent, main points, and action items.
- **Hugging Face:** provides the API client/router and uses `hf-inference` for ASR and `featherless-ai` for the default chat model. One `HF_TOKEN` is used for both provider calls.
- **Pydantic:** validates the model's JSON against the application schema. Both provider paths use JSON Object Mode with an explicit JSON-only prompt, followed by local Pydantic validation.

## Boundaries

- The app rejects files larger than 25 MB before upload; provider limits can change.
- The tool does not identify speakers unless the transcript itself contains speaker information.
- Hugging Face ASR currently uses Whisper automatic language detection; the optional language hint is applied only by the Groq route.
- Due dates and owners are `null`/unknown when they are not stated; the model is instructed not to invent them.
- Audio and transcripts are sent to the selected provider when the API path is used. Do not use confidential recordings without reviewing current provider terms.
