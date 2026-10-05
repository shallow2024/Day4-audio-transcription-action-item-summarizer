"""Transcribe short audio with a hosted ASR model and extract meeting actions.

The module keeps provider calls behind a small protocol so all parsing and
Markdown/JSON rendering can be tested without an API key.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

DEFAULT_PROVIDER = "huggingface"
DEFAULT_TRANSCRIPTION_MODEL = "openai/whisper-large-v3"
DEFAULT_SUMMARY_MODEL = "Qwen/Qwen2.5-72B-Instruct"
GROQ_TRANSCRIPTION_MODEL = "whisper-large-v3"
GROQ_SUMMARY_MODEL = "llama-3.3-70b-versatile"
HF_ASR_PROVIDER = "hf-inference"
HF_CHAT_PROVIDER = "featherless-ai"
MAX_AUDIO_BYTES = 25 * 1024 * 1024
SUPPORTED_EXTENSIONS = {".flac", ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".ogg", ".wav", ".webm"}


class AudioSummarizerError(RuntimeError):
    """Expected, user-facing error."""


class ActionItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str = Field(min_length=1)
    owner: str | None = None
    due_date: str | None = None
    priority: str = "medium"


class SummaryReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    speaker_intent: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    main_points: list[str] = Field(min_length=1)
    action_items: list[ActionItem]


class ProviderClient(Protocol):
    def transcribe(self, audio_path: Path, language: str | None = None) -> str: ...

    def summarize(self, transcript: str) -> SummaryReport: ...


def _summary_system_prompt() -> str:
    schema_description = {
        "speaker_intent": "short description of why the speaker is talking",
        "summary": "one concise paragraph",
        "main_points": ["important point"],
        "action_items": [
            {"task": "specific next step", "owner": "person or null", "due_date": "date or null", "priority": "high|medium|low"}
        ],
    }
    return (
        "You extract meeting or voice-note actions. Respond with JSON only, no Markdown and no commentary. "
        "Use exactly these keys and types. Do not invent facts; use null for unknown owner or due date. "
        f"Schema example: {json.dumps(schema_description)}"
    )


def _parse_summary_response(content: str | None, provider_name: str) -> SummaryReport:
    if not content:
        raise AudioSummarizerError(f"{provider_name} returned an empty summary.")
    try:
        return SummaryReport.model_validate_json(content)
    except (ValidationError, ValueError) as exc:
        raise AudioSummarizerError(f"The model returned invalid summary JSON: {exc}") from exc


class GroqProvider:
    def __init__(self, transcription_model: str = DEFAULT_TRANSCRIPTION_MODEL, summary_model: str = DEFAULT_SUMMARY_MODEL) -> None:
        try:
            from groq import Groq
        except ImportError as exc:
            raise AudioSummarizerError("Install dependencies first: py -m pip install -r requirements.txt") from exc
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise AudioSummarizerError("GROQ_API_KEY is missing. Set it in PowerShell before making an API request.")
        self.client = Groq(api_key=api_key)
        self.transcription_model = transcription_model
        self.summary_model = summary_model

    def transcribe(self, audio_path: Path, language: str | None = None) -> str:
        try:
            with audio_path.open("rb") as audio_file:
                kwargs: dict[str, Any] = {
                    "file": audio_file,
                    "model": self.transcription_model,
                    "response_format": "json",
                    "temperature": 0.0,
                }
                if language:
                    kwargs["language"] = language
                response = self.client.audio.transcriptions.create(**kwargs)
        except Exception as exc:
            raise AudioSummarizerError(f"Groq transcription failed: {exc}") from exc
        text = getattr(response, "text", None)
        if not text or not text.strip():
            raise AudioSummarizerError("Groq returned an empty transcript.")
        return text.strip()

    def summarize(self, transcript: str) -> SummaryReport:
        try:
            response = self.client.chat.completions.create(
                model=self.summary_model,
                temperature=0,
                max_tokens=2_000,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _summary_system_prompt()},
                    {"role": "user", "content": f"Transcript:\n\n{transcript}"},
                ],
            )
        except Exception as exc:
            raise AudioSummarizerError(f"Groq summarization failed: {exc}") from exc
        content = response.choices[0].message.content if response.choices else None
        return _parse_summary_response(content, "Groq")


class HuggingFaceProvider:
    """Use HF Inference Providers: hf-inference for ASR and Featherless for chat."""

    def __init__(self, transcription_model: str = DEFAULT_TRANSCRIPTION_MODEL, summary_model: str = DEFAULT_SUMMARY_MODEL) -> None:
        try:
            from huggingface_hub import InferenceClient
        except ImportError as exc:
            raise AudioSummarizerError("Install dependencies first: py -m pip install -r requirements.txt") from exc
        token = os.getenv("HF_TOKEN")
        if not token:
            raise AudioSummarizerError("HF_TOKEN is missing. Set it in PowerShell before using --provider huggingface.")
        self.asr = InferenceClient(provider=HF_ASR_PROVIDER, token=token)
        self.chat = InferenceClient(provider=HF_CHAT_PROVIDER, token=token)
        self.transcription_model = transcription_model
        self.summary_model = summary_model

    def transcribe(self, audio_path: Path, language: str | None = None) -> str:
        try:
            extra_body = {"language": language} if language else None
            result = self.asr.automatic_speech_recognition(audio_path, model=self.transcription_model, extra_body=extra_body)
        except Exception as exc:
            raise AudioSummarizerError(f"Hugging Face transcription failed: {exc}") from exc
        text = getattr(result, "text", None)
        if not text or not text.strip():
            raise AudioSummarizerError("Hugging Face returned an empty transcript.")
        return text.strip()

    def summarize(self, transcript: str) -> SummaryReport:
        try:
            response = self.chat.chat_completion(
                model=self.summary_model,
                messages=[
                    {"role": "system", "content": _summary_system_prompt()},
                    {"role": "user", "content": f"Transcript:\n\n{transcript}"},
                ],
                temperature=0,
                max_tokens=2_000,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            raise AudioSummarizerError(f"Hugging Face summarization failed: {exc}") from exc
        choices = getattr(response, "choices", [])
        content = getattr(choices[0].message, "content", None) if choices else None
        return _parse_summary_response(content, "Hugging Face")


def validate_audio_path(audio_path: Path, max_bytes: int = MAX_AUDIO_BYTES) -> Path:
    path = audio_path.expanduser().resolve()
    if not path.is_file():
        raise AudioSummarizerError(f"Audio file was not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise AudioSummarizerError(f"Unsupported audio extension {path.suffix!r}. Supported: {allowed}")
    size = path.stat().st_size
    if size == 0:
        raise AudioSummarizerError("The audio file is empty.")
    if size > max_bytes:
        raise AudioSummarizerError(f"Audio file is too large ({size:,} bytes). Maximum is {max_bytes:,} bytes.")
    return path


def render_markdown(report: SummaryReport, transcript: str, source: Path) -> str:
    lines = [
        f"# Audio Action-Item Report\n",
        f"**Source:** `{source.name}`\n",
        f"## Speaker intent\n\n{report.speaker_intent}\n",
        f"## Summary\n\n{report.summary}\n",
        "## Main points\n",
    ]
    lines.extend(f"- {point}" for point in report.main_points)
    lines.append("\n## Action items\n")
    if report.action_items:
        lines.append("| Task | Owner | Due date | Priority |\n|---|---|---|---|")
        lines.extend(
            f"| {item.task} | {item.owner or 'Unknown'} | {item.due_date or 'Unknown'} | {item.priority} |"
            for item in report.action_items
        )
    else:
        lines.append("No explicit action items were identified.")
    lines.append(f"\n## Transcript\n\n{transcript}\n")
    return "\n".join(lines)


def build_report(audio_path: Path, provider: ProviderClient, language: str | None = None) -> tuple[str, SummaryReport]:
    path = validate_audio_path(audio_path)
    transcript = provider.transcribe(path, language)
    if not transcript.strip():
        raise AudioSummarizerError("The transcript is empty.")
    report = provider.summarize(transcript)
    return transcript, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe audio and extract structured action items with an AI provider.")
    parser.add_argument("audio", type=Path, help="MP3, WAV, M4A, OGG, FLAC, MP4, WEBM, or related supported audio file")
    parser.add_argument("--language", help="Optional ISO-639-1 language code, such as en or zh")
    parser.add_argument("--output", type=Path, help="Optional output path; .json writes JSON, other extensions write Markdown")
    parser.add_argument("--provider", choices=("huggingface", "groq"), default=DEFAULT_PROVIDER)
    parser.add_argument("--transcription-model", help="Provider-specific transcription model")
    parser.add_argument("--summary-model", help="Provider-specific summarization model")
    parser.add_argument("--transcript-only", action="store_true", help="Print only the transcript and skip summarization")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        path = validate_audio_path(args.audio)
        if args.provider == "huggingface":
            provider = HuggingFaceProvider(args.transcription_model or DEFAULT_TRANSCRIPTION_MODEL, args.summary_model or DEFAULT_SUMMARY_MODEL)
        else:
            provider = GroqProvider(args.transcription_model or GROQ_TRANSCRIPTION_MODEL, args.summary_model or GROQ_SUMMARY_MODEL)
        transcript = provider.transcribe(path, args.language)
        if args.transcript_only:
            print(transcript)
            return 0
        report = provider.summarize(transcript)
        payload = report.model_dump()
        if args.output and args.output.suffix.lower() == ".json":
            args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"Wrote JSON report: {args.output}")
        elif args.output:
            args.output.write_text(render_markdown(report, transcript, path), encoding="utf-8")
            print(f"Wrote Markdown report: {args.output}")
        else:
            print(render_markdown(report, transcript, path))
        return 0
    except AudioSummarizerError as exc:
        print(f"Error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
