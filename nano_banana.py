"""
Nano Banana Pro / Nano Banana Video API client.
Set NANO_BANANA_API_KEY. Same-style options as Veo: prompt, duration, resolution, aspect_ratio.
"""
import os
import time
from pathlib import Path

BASE_URL = "https://nanobananavideo.com/api/v1"


def generate_video(
    prompt: str,
    output_path: str,
    *,
    duration_seconds: int = 5,
    resolution: str = "720p",
    aspect_ratio: str = "16:9",
) -> str:
    """
    Generate video via Nano Banana API; poll until done and download to output_path.
    """
    api_key = os.environ.get("NANO_BANANA_API_KEY", "").strip()
    if not api_key:
        raise ValueError("NANO_BANANA_API_KEY is required for Nano Banana Pro.")

    # Clamp duration to API range 3–12
    duration_seconds = max(3, min(12, duration_seconds))
    # Map resolution
    if resolution == "4k":
        resolution = "1080p"
    resolution = "480p" if resolution == "480p" else "720p" if resolution == "720p" else "1080p"

    headers = {
        "Content-Type": "application/json",
        "X-API-Key": api_key,
    }
    payload = {
        "prompt": prompt[:500],
        "duration": duration_seconds,
        "resolution": resolution,
        "aspect_ratio": aspect_ratio,
    }

    try:
        import urllib.request
        import json

        req = urllib.request.Request(
            f"{BASE_URL}/text-to-video.php",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        raise RuntimeError(f"Nano Banana API request failed: {e}") from e

    if not data.get("success"):
        raise RuntimeError(data.get("error", "Unknown Nano Banana API error"))

    video_id = data.get("video_id")
    video_url = data.get("video_url")
    if video_url:
        # Synchronous response with URL
        _download_url(video_url, output_path)
        return output_path

    if not video_id:
        raise RuntimeError("Nano Banana API did not return video_id or video_url")

    # Poll status
    for _ in range(120):
        time.sleep(5)
        try:
            status_req = urllib.request.Request(
                f"{BASE_URL}/video-status.php?video_id={video_id}",
                headers={"X-API-Key": api_key},
                method="GET",
            )
            with urllib.request.urlopen(status_req, timeout=30) as resp:
                status_data = json.loads(resp.read().decode("utf-8"))
        except Exception:
            continue
        if status_data.get("status") == "completed" and status_data.get("video_url"):
            _download_url(status_data["video_url"], output_path)
            return output_path
        if status_data.get("status") == "failed":
            raise RuntimeError(status_data.get("error", "Nano Banana generation failed"))

    raise RuntimeError("Nano Banana generation timed out.")


def _download_url(url: str, path: str) -> None:
    import urllib.request
    urllib.request.urlretrieve(url, path)
