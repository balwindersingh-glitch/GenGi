"""
Analyze an uploaded video with Gemini and return per-segment descriptions
plus replication prompts for use with Veo + character reference.
Uses same GCP credentials as Veo (Vertex AI).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, List

from generate_video import get_config


ANALYZE_PROMPT = """You are an expert video and audio analyst. The user will replicate this video with Google Veo using a character reference image. Your job is to write extremely detailed prompts and to describe everything that is HEARD as well as seen. The video has an audio track—listen to it carefully.

Break the video into fixed 8-second segments: 0–8s, 8–16s, 16–24s, 24–32s, and so on (each segment is exactly 8 seconds). For each segment provide:

- start_sec: start time in seconds (0, 8, 16, 24, 32, ...)
- end_sec: end time in seconds (8, 16, 24, 32, 40, ...)—always start_sec + 8
- description: a concise summary of the segment (action + key audio, for quick scanning).
- singing: What the person is singing or saying in this segment. If they sing: transcribe the lyrics if you can make them out (even partially), or describe the vocal (e.g. "singing a slow melody in English, emotional chorus", "humming then words in second half"). If they speak: quote or summarize. Language, style (ballad, pop, rap, spoken), and mood. If no singing/speech in this segment, use empty string "".
- background_sounds: Everything else on the audio track. Music: genre, instruments (piano, guitar, drums, synth), tempo, volume (foreground vs background). Ambient: crowd, traffic, nature, room tone, wind, room echo. Sound effects: doors, footsteps, etc. If none, use "".
- prompt: a very long, exhaustive VISUAL prompt (5–12 sentences). Describe every visible detail so the AI can recreate the image. Include:

  1. CHARACTER AND BODY: Exact pose, posture, every visible movement, facial expression, gaze, speed of movements, gestures, interaction with objects.

  2. CAMERA: Shot size, angle, movement, framing, depth of field.

  3. LIGHTING: Quality, direction, source, color temperature, shadows, highlights, exposure.

  4. ENVIRONMENT: Location, details, background, foreground, props, colors, atmosphere.

  5. PACING AND MOTION: Static/slow/dynamic, rhythm, motion blur.

  6. STYLE: Look and feel, mood, lens.

Write the prompt in continuous prose. Use "The character" or "A person". At the end of the prompt you may add one sentence about audio if relevant for context (e.g. "The character appears to be singing; lips moving in sync with emotional delivery.") so the visual matches the sound.

Return ONLY a valid JSON array of objects with keys: start_sec, end_sec, description, prompt, singing, background_sounds. Use empty string "" for singing and background_sounds when there is nothing to report. No markdown, no code fence, no explanation—just the raw JSON array."""


def analyze_video_for_prompts(video_path: str) -> List[dict[str, Any]]:
    """
    Analyze video with Gemini and return list of segments with descriptions and replication prompts.
    Each item: {"start_sec": float, "end_sec": float, "description": str, "prompt": str}
    """
    from google import genai
    from google.genai import types

    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Limit size for inline upload (e.g. 50MB); larger files would need File API
    max_bytes = 50 * 1024 * 1024
    video_bytes = path.read_bytes()
    if len(video_bytes) > max_bytes:
        raise ValueError(
            f"Video too large ({len(video_bytes) / 1024 / 1024:.1f} MB). Max {max_bytes // 1024 // 1024} MB for analysis."
        )

    cfg = get_config()
    project_id = cfg["project_id"]
    if not project_id:
        raise ValueError("Project ID required. Set GOOGLE_CLOUD_PROJECT or use GCP credentials.")

    location = cfg["location"]
    client = genai.Client(vertexai=True, project=project_id, location=location)

    # Prefer a model that supports video; gemini-2.0-flash or 2.5-flash
    model_id = os.environ.get("GEMINI_ANALYZE_MODEL", "gemini-2.0-flash")

    video_part = types.Part.from_bytes(data=video_bytes, mime_type="video/mp4")
    response = client.models.generate_content(
        model=model_id,
        contents=[video_part, ANALYZE_PROMPT],
    )

    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("Gemini returned no text.")

    # Extract JSON array (in case model wrapped in markdown)
    text = text.strip()
    if "```json" in text:
        text = re.sub(r"^.*?```json\s*", "", text)
        text = re.sub(r"\s*```.*$", "", text)
    elif "```" in text:
        text = re.sub(r"^.*?```\s*", "", text)
        text = re.sub(r"\s*```.*$", "", text)
    text = text.strip()

    try:
        segments = json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Could not parse Gemini response as JSON: {e}") from e

    if not isinstance(segments, list):
        raise RuntimeError("Response is not a JSON array.")

    out = []
    for i, s in enumerate(segments):
        if not isinstance(s, dict):
            continue
        out.append({
            "start_sec": float(s.get("start_sec", i * 6)),
            "end_sec": float(s.get("end_sec", (i + 1) * 6)),
            "description": str(s.get("description", "")),
            "prompt": str(s.get("prompt", "")),
            "singing": str(s.get("singing", "")),
            "background_sounds": str(s.get("background_sounds", "")),
        })
    return out
