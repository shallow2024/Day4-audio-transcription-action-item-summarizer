# Architecture

```mermaid
flowchart LR
    A[Local MP3/WAV/audio file] --> B[Validate extension and 25 MB limit]
    B --> C[Groq Whisper whisper-large-v3]
    C --> D[Transcript text]
    D --> E[Groq llama-3.3-70b-versatile]
    E --> F[JSON object mode]
    F --> G[Pydantic validation]
    G --> H[JSON or Markdown report]
```

## Responsibilities

- **Local Python:** validates the file, opens the upload, orchestrates the two API calls, validates the result, and renders output.
- **Whisper:** converts speech to a transcript. It does not produce action items.
- **Llama 3.3:** reads the transcript and extracts intent, main points, and action items.
- **Pydantic:** validates the model's JSON against the application schema. The Groq `llama-3.3-70b-versatile` path uses JSON Object Mode with an explicit JSON-only prompt, followed by local Pydantic validation.

## Boundaries

- Groq's current direct upload limit is documented as 25 MB for the free tier; the app rejects larger files before upload.
- The tool does not identify speakers unless the transcript itself contains speaker information.
- Due dates and owners are `null`/unknown when they are not stated; the model is instructed not to invent them.
- Audio and transcripts are sent to Groq when the API path is used. Do not use confidential recordings without reviewing current provider terms.
