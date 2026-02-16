"""
Nano Banana image generation via Vertex AI (Gemini).
Uses the same GCP project and credentials as Veo — no separate API key.
See: https://ai.google.dev/gemini-api/docs/image-generation
"""
import os
from pathlib import Path
from typing import List, Optional, Tuple

from generate_video import get_config


def generate_image(
    prompt: str,
    output_path: str,
    *,
    aspect_ratio: str = "16:9",
    resolution: str = "1K",
    negative_prompt: Optional[str] = None,
    reference_images: Optional[List[Tuple[bytes, str]]] = None,
    model_id: Optional[str] = None,
) -> str:
    """
    Generate an image with Gemini (Nano Banana) on Vertex AI.
    Uses same GOOGLE_APPLICATION_CREDENTIALS / project as Veo.
    resolution: "1K", "2K", "4K" (Gemini 3 Pro); "1K" or default for 2.5 Flash.
    """
    from google import genai
    from google.genai import types

    cfg = get_config()
    project_id = cfg["project_id"]
    if not project_id:
        raise ValueError(
            "Project ID required. Set GOOGLE_CLOUD_PROJECT or GCP_PROJECT_ID (or use same credentials as Veo)."
        )
    model_id = model_id or os.environ.get("NANO_BANANA_IMAGE_MODEL", "gemini-2.5-flash-image")
    # Gemini 3 Pro Image is only available in region "global"; Flash uses regional (e.g. us-central1)
    location = "global" if "gemini-3-pro-image" in model_id else cfg["location"]

    # Combine prompt with avoid/negative guidance (semantic negative prompt)
    full_prompt = prompt
    if negative_prompt and negative_prompt.strip():
        full_prompt = f"{prompt.strip()}\n\nAvoid: {negative_prompt.strip()}"

    client = genai.Client(vertexai=True, project=project_id, location=location)

    # ImageConfig only accepts aspect_ratio in this SDK; image_size (1K/2K/4K) is not supported here
    image_config = types.ImageConfig(aspect_ratio=aspect_ratio)

    config = types.GenerateContentConfig(
        response_modalities=["TEXT", "IMAGE"],
        image_config=image_config,
    )

    # Build contents: optional reference images + text prompt (image-to-image or edit)
    if reference_images:
        content_parts = []
        for img_bytes, mime_type in reference_images:
            try:
                part = types.Part.from_bytes(data=img_bytes, mime_type=mime_type)
            except (AttributeError, TypeError):
                blob = types.Blob(data=img_bytes, mime_type=mime_type)
                part = types.Part(inline_data=blob)
            content_parts.append(part)
        try:
            content_parts.append(types.Part.from_text(text=full_prompt))
        except (AttributeError, TypeError):
            content_parts.append(types.Part(text=full_prompt))
        contents = content_parts
    else:
        contents = full_prompt

    response = client.models.generate_content(
        model=model_id,
        contents=contents,
        config=config,
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out_path = str(path) if path.suffix else str(path) + ".png"
    parts = []
    if getattr(response, "candidates", None) and len(response.candidates) > 0:
        c = response.candidates[0]
        if getattr(c, "content", None) and getattr(c.content, "parts", None):
            parts = c.content.parts
    if not parts:
        parts = getattr(response, "parts", [])
    for part in parts or []:
        if getattr(part, "inline_data", None):
            idata = part.inline_data
            data = getattr(idata, "data", None)
            if data:
                with open(out_path, "wb") as f:
                    f.write(data)
                return out_path
        if getattr(part, "as_image", None):
            try:
                img = part.as_image()
                if img is not None:
                    img.save(out_path)
                    return out_path
            except Exception:
                pass
    raise RuntimeError("No image in Gemini response. Try a different prompt.")
