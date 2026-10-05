from pathlib import Path

import pytest
from pydantic import ValidationError

from audio_summarizer import (
    ActionItem,
    AudioSummarizerError,
    SummaryReport,
    render_markdown,
    validate_audio_path,
)


def test_validate_supported_audio(tmp_path: Path):
    audio = tmp_path / "note.wav"
    audio.write_bytes(b"RIFF demo")
    assert validate_audio_path(audio) == audio.resolve()


def test_validate_rejects_missing_file(tmp_path: Path):
    with pytest.raises(AudioSummarizerError, match="not found"):
        validate_audio_path(tmp_path / "missing.wav")


def test_validate_rejects_extension(tmp_path: Path):
    file = tmp_path / "note.txt"
    file.write_text("not audio", encoding="utf-8")
    with pytest.raises(AudioSummarizerError, match="Unsupported"):
        validate_audio_path(file)


def test_validate_rejects_large_file(tmp_path: Path):
    file = tmp_path / "note.mp3"
    file.write_bytes(b"x" * 11)
    with pytest.raises(AudioSummarizerError, match="too large"):
        validate_audio_path(file, max_bytes=10)


def test_report_validation_and_markdown(tmp_path: Path):
    source = tmp_path / "meeting.wav"
    report = SummaryReport(
        speaker_intent="Coordinate the release.",
        summary="The team discussed the release timeline.",
        main_points=["Release is planned for Friday."],
        action_items=[ActionItem(task="Confirm deployment checklist", owner="Mina", due_date="Friday", priority="high")],
    )
    markdown = render_markdown(report, "We will release on Friday.", source)
    assert "# Audio Action-Item Report" in markdown
    assert "Confirm deployment checklist" in markdown
    assert "Mina" in markdown


def test_report_rejects_extra_fields():
    with pytest.raises(ValidationError):
        SummaryReport.model_validate({
            "speaker_intent": "Discuss",
            "summary": "Summary",
            "main_points": ["Point"],
            "action_items": [],
            "unexpected": True,
        })


class FakeProvider:
    def transcribe(self, audio_path: Path, language: str | None = None) -> str:
        return "Please send the draft by Friday."

    def summarize(self, transcript: str) -> SummaryReport:
        return SummaryReport(
            speaker_intent="Request a draft.",
            summary="A draft is requested by Friday.",
            main_points=["Draft requested."],
            action_items=[ActionItem(task="Send the draft", due_date="Friday")],
        )


def test_fake_provider_contract(tmp_path: Path):
    audio = tmp_path / "note.wav"
    audio.write_bytes(b"demo")
    provider = FakeProvider()
    transcript = provider.transcribe(audio)
    report = provider.summarize(transcript)
    assert report.action_items[0].task == "Send the draft"
