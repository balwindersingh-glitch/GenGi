"""
Veo video generation using Vertex AI (Google Gen AI SDK).
Configure via environment variables; run with CLI args for prompt, output, and options.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from typing import List, Optional, Tuple

from google import genai
from google.genai import types


def _is_provisioning_error(err: object) -> bool:
    """True if error is Vertex 'service agents are being provisioned' (code 9)."""
    if hasattr(err, "code") and getattr(err, "code", None) == 9:
        return True
    if isinstance(err, dict) and err.get("code") == 9:
        return True
    msg = (getattr(err, "message", None) or (err.get("message") if isinstance(err, dict) else None) or str(err))
    return "service agents" in str(msg).lower() or "provisioning" in str(msg).lower()


def _project_id_from_credentials() -> str:
    """Read project_id from the JSON key file if GOOGLE_APPLICATION_CREDENTIALS is set."""
    path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not path or not os.path.isfile(path):
        return ""
    try:
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("project_id", "")
    except Exception:
        return ""


def get_config() -> dict:
    """Read config from environment with defaults. Project ID can come from env or from the credentials JSON."""
    project_id = (
        os.environ.get("GOOGLE_CLOUD_PROJECT")
        or os.environ.get("GCP_PROJECT_ID")
        or _project_id_from_credentials()
        or ""
    )
    return {
        "project_id": project_id,
        "location": os.environ.get("GOOGLE_CLOUD_LOCATION") or os.environ.get("VERTEX_LOCATION", "us-central1"),
        "model_id": os.environ.get("VEO_MODEL_ID", "veo-3.1-generate-001"),
        "output_gcs_uri": os.environ.get("VEO_OUTPUT_GCS_URI") or None,
    }


def generate_video(
    prompt_text: str,
    output_file: str = "output_video.mp4",
    *,
    project_id: Optional[str] = None,
    location: Optional[str] = None,
    model_id: Optional[str] = None,
    output_gcs_uri: Optional[str] = None,
    aspect_ratio: str = "16:9",
    duration_seconds: int = 6,
    number_of_videos: int = 1,
    resolution: str = "720p",
    generate_audio: bool = False,
    person_generation: str = "allow_adult",
    negative_prompt: Optional[str] = None,
    seed: Optional[int] = None,
    reference_images: Optional[List[Tuple[bytes, str]]] = None,
    first_frame: Optional[Tuple[bytes, str]] = None,
    last_frame: Optional[Tuple[bytes, str]] = None,
) -> str:
    """
    Sends a prompt to the Veo API and saves the resulting video.
    Uses a long-running operation; polls every 15s until done.
    Returns the path to the saved file.
    """
    cfg = get_config()
    project_id = project_id or cfg["project_id"]
    location = location or cfg["location"]
    model_id = model_id or cfg["model_id"]
    output_gcs_uri = output_gcs_uri if output_gcs_uri is not None else cfg["output_gcs_uri"]

    if not project_id:
        raise ValueError(
            "Project ID is required. Set GOOGLE_CLOUD_PROJECT or GCP_PROJECT_ID, or pass project_id."
        )

    preview = prompt_text[:80] + "..." if len(prompt_text) > 80 else prompt_text
    print(f"Generating video for prompt: '{preview}'")

    client = genai.Client(vertexai=True, project=project_id, location=location)

    config_kw: dict = {
        "aspect_ratio": aspect_ratio,
        "number_of_videos": number_of_videos,
        "duration_seconds": duration_seconds,
        "resolution": resolution,
        "person_generation": person_generation,
        "generate_audio": generate_audio,
    }
    if output_gcs_uri:
        config_kw["output_gcs_uri"] = output_gcs_uri
    if seed is not None:
        config_kw["seed"] = seed
    if negative_prompt:
        config_kw["negative_prompt"] = negative_prompt.strip()
    if reference_images and len(reference_images) > 0:
        config_kw["reference_images"] = [
            types.VideoGenerationReferenceImage(
                image=types.Image(image_bytes=img_bytes, mime_type=mime_type),
                reference_type=types.VideoGenerationReferenceType.ASSET,
            )
            for img_bytes, mime_type in reference_images[:3]
        ]
    if last_frame:
        last_bytes, last_mime = last_frame
        config_kw["last_frame"] = types.Image(image_bytes=last_bytes, mime_type=last_mime)

    call_kw: dict = {
        "model": model_id,
        "prompt": prompt_text,
        "config": types.GenerateVideosConfig(**config_kw),
    }
    if first_frame:
        first_bytes, first_mime = first_frame
        call_kw["image"] = types.Image(image_bytes=first_bytes, mime_type=first_mime)

    operation = client.models.generate_videos(**call_kw)

    while not operation.done:
        time.sleep(15)
        operation = client.operations.get(operation)
        print("  ... still generating (polling)")

    if not operation.response:
        err = getattr(operation, "error", None)
        err_str = str(err) if err else ""
        if err and (_is_provisioning_error(err) or "service agents" in err_str.lower() or "provisioning" in err_str.lower()):
            raise RuntimeError(
                "Vertex AI is still provisioning service agents for your project. "
                "This usually takes a few minutes after first use or when using a new bucket. "
                "Please try again in 5–10 minutes. See: https://cloud.google.com/vertex-ai/docs/general/access-control#service-agents"
            )
        msg = "Video generation failed or returned no response."
        if err:
            msg += " " + err_str
        raise RuntimeError(msg)

    # SDK may expose result on .result or .response; videos as .generated_videos or .videos
    result = getattr(operation, "result", None) or operation.response
    generated = []
    if result:
        generated = getattr(result, "generated_videos", None) or getattr(result, "videos", None) or []
    if not generated:
        err = getattr(operation, "error", None)
        detail = str(err) if err else ""
        if err and (_is_provisioning_error(err) or "service agents" in detail.lower() or "provisioning" in detail.lower()):
            raise RuntimeError(
                "Vertex AI is still provisioning service agents for your project. "
                "This usually takes a few minutes after first use or when using a new bucket. "
                "Please try again in 5–10 minutes. See: https://cloud.google.com/vertex-ai/docs/general/access-control#service-agents"
            )
        rai = getattr(result, "rai_media_filtered_count", None) if result else None
        if rai is not None and int(rai) > 0:
            detail = (detail + " " if detail else "") + f"Content filtered (RAI count: {rai})."
        if not detail:
            detail = " Output may have been blocked by safety filters—try a different prompt or avoid people/faces if not allowed."
        raise RuntimeError("No generated videos in response." + detail)

    # First item may be the video object or have a .video attribute
    first = generated[0]
    video = getattr(first, "video", first)

    if getattr(video, "video_bytes", None):
        video_bytes = video.video_bytes
        with open(output_file, "wb") as f:
            f.write(video_bytes)
        print(f"Video saved successfully to {output_file}")
        return output_file

    video_uri = getattr(video, "uri", None)
    if video_uri:
        try:
            from google.cloud import storage
        except ImportError:
            raise ImportError(
                "Output was written to GCS. Install google-cloud-storage and run again, "
                "or leave VEO_OUTPUT_GCS_URI unset to get in-memory video_bytes."
            ) from None
        path = video_uri.replace("gs://", "", 1)
        bucket_name, _, object_name = path.partition("/")
        client_storage = storage.Client(project=project_id)
        bucket = client_storage.bucket(bucket_name)
        blob = bucket.blob(object_name)
        blob.download_to_filename(output_file)
        print(f"Video downloaded from GCS to {output_file}")
        return output_file

    raise RuntimeError("Response contained no video bytes or GCS URI.")


# Extend-capable models (per Vertex AI docs)
EXTEND_MODEL_IDS = ("veo-3.1-generate-preview", "veo-3.1-fast-generate-preview")


def extend_video(
    input_video_path: str,
    prompt_text: str,
    output_file: str = "extended_video.mp4",
    *,
    project_id: Optional[str] = None,
    location: Optional[str] = None,
    model_id: str = "veo-3.1-generate-preview",
    aspect_ratio: str = "16:9",
    output_gcs_uri: Optional[str] = None,
    generate_audio: bool = False,
) -> str:
    """
    Extend a Veo-generated video by 7 seconds. Input must be 1–30 sec, 24fps, 720p or 1080p, MP4.
    Output is the original + 7 seconds (single merged file), 720p.
    Uses veo-3.1-generate-preview or veo-3.1-fast-generate-preview.
    Note: Vertex extend API is currently video-only for the new 7s; generate_audio is passed in case the API adds audio support.
    """
    from pathlib import Path

    path = Path(input_video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {input_video_path}")
    if not prompt_text or not prompt_text.strip():
        raise ValueError("Prompt is required for extend (describe what happens in the next 7 seconds).")

    cfg = get_config()
    project_id = project_id or cfg["project_id"]
    location = location or cfg["location"]
    output_gcs_uri = output_gcs_uri if output_gcs_uri is not None else cfg.get("output_gcs_uri")
    if not project_id:
        raise ValueError("Project ID is required.")
    if not output_gcs_uri or not str(output_gcs_uri).strip():
        raise ValueError(
            "Extend requires VEO_OUTPUT_GCS_URI. Set it to a GCS path (e.g. gs://your-bucket/veo-output/). "
            "The extended video is written there then downloaded."
        )
    output_gcs_uri = str(output_gcs_uri).rstrip("/")
    if model_id not in EXTEND_MODEL_IDS:
        model_id = EXTEND_MODEL_IDS[0]

    # Vertex extend requires the input video as a GCS URI (allowlisted project), not raw bytes.
    path_no_gs = output_gcs_uri.replace("gs://", "", 1)
    bucket_name, _, prefix = path_no_gs.partition("/")
    input_object = f"{prefix.rstrip('/')}/input_extend_{uuid.uuid4().hex}.mp4"
    input_gcs_uri = f"gs://{bucket_name}/{input_object}"
    try:
        from google.cloud import storage
    except ImportError:
        raise ImportError("Extend with GCS requires google-cloud-storage.") from None
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(input_object)
    blob.upload_from_filename(str(path), content_type="video/mp4")
    try:
        video_input = types.Video(uri=input_gcs_uri, mime_type="video/mp4")
        client = genai.Client(vertexai=True, project=project_id, location=location)
        config_kw: dict = {
            "aspect_ratio": aspect_ratio,
            "number_of_videos": 1,
            "output_gcs_uri": output_gcs_uri,
            "generate_audio": generate_audio,
        }
        operation = client.models.generate_videos(
            model=model_id,
            prompt=prompt_text.strip(),
            video=video_input,
            config=types.GenerateVideosConfig(**config_kw),
        )

        while not operation.done:
            time.sleep(15)
            operation = client.operations.get(operation)
            print("  ... extending (polling)")

        if not operation.response:
            err = getattr(operation, "error", None)
            err_str = str(err) if err else ""
            err_lower = err_str.lower()
            if err and (_is_provisioning_error(err) or "service agents" in err_lower or "provisioning" in err_lower):
                raise RuntimeError(
                    "Vertex AI is still provisioning service agents for your project. "
                    "This usually takes a few minutes after first use or when using a new bucket. "
                    "Please try again in 5–10 minutes. See: https://cloud.google.com/vertex-ai/docs/general/access-control#service-agents"
                )
            if "allowlist" in err_lower or "not allowlisted" in err_lower:
                raise RuntimeError(
                    "Your project is not allowlisted for Veo video extend (video as input). "
                    "The input video was uploaded to your GCS bucket and passed by URI. "
                    "To use extend, your Google Cloud project may need to be allowlisted for this feature. "
                    "Contact Google Cloud support or your account team, or see: "
                    "https://cloud.google.com/vertex-ai/generative-ai/docs/video/extend-a-veo-video"
                )
            raise RuntimeError(f"Extend failed: {err or 'No response'}")

        result = getattr(operation, "result", None) or operation.response
        generated = getattr(result, "generated_videos", None) or getattr(result, "videos", None) or []
        if not generated:
            raise RuntimeError("No extended video in response.")
        first = generated[0]
        video = getattr(first, "video", first)
        if getattr(video, "video_bytes", None):
            with open(output_file, "wb") as f:
                f.write(video.video_bytes)
            print(f"Extended video saved to {output_file}")
            return output_file
        video_uri = getattr(video, "uri", None)
        if video_uri:
            path_ = video_uri.replace("gs://", "", 1)
            out_bucket_name, _, object_name = path_.partition("/")
            client_storage = storage.Client(project=project_id)
            out_bucket = client_storage.bucket(out_bucket_name)
            out_blob = out_bucket.blob(object_name)
            out_blob.download_to_filename(output_file)
            print(f"Extended video downloaded to {output_file}")
            return output_file
        raise RuntimeError("Response contained no video bytes or GCS URI.")
    finally:
        try:
            blob.delete()
        except Exception:
            pass


def _read_prompt(prompt: Optional[str], prompt_file: Optional[str]) -> str:
    if prompt_file:
        path = os.path.expanduser(prompt_file)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Prompt file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    if prompt:
        return prompt.strip()
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    raise ValueError("Provide --prompt, --prompt-file, or pipe prompt via stdin.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate video from a text prompt using Vertex AI Veo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        help="Text prompt for video generation.",
    )
    parser.add_argument(
        "--prompt-file", "-f",
        type=str,
        metavar="PATH",
        help="Read prompt from a text file.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output_video.mp4",
        metavar="FILE",
        help="Output video file path.",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=os.environ.get("VEO_MODEL_ID", "veo-3.1-generate-001"),
        help="Veo model ID (e.g. veo-3.1-generate-001, veo-3.1-fast-generate-001).",
    )
    parser.add_argument(
        "--aspect-ratio",
        type=str,
        default="16:9",
        choices=("16:9", "9:16"),
        help="Aspect ratio of the generated video.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=6,
        choices=(4, 6, 8),
        metavar="SEC",
        help="Video duration in seconds.",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="720p",
        choices=("720p", "1080p", "4k"),
        help="Output resolution (4k only for some preview models).",
    )
    parser.add_argument(
        "--audio",
        action="store_true",
        help="Generate audio for the video.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="N",
        help="Seed for deterministic generation (0–4294967295).",
    )
    parser.add_argument(
        "--gcs-uri",
        type=str,
        default=os.environ.get("VEO_OUTPUT_GCS_URI"),
        metavar="URI",
        help="GCS URI for output (e.g. gs://bucket/prefix/). If unset, video is returned in-memory.",
    )
    parser.add_argument(
        "--person-generation",
        type=str,
        default="allow_adult",
        choices=("allow_adult", "dont_allow", "allow_all"),
        help="Person/face generation policy.",
    )
    args = parser.parse_args()

    try:
        prompt_text = _read_prompt(args.prompt, args.prompt_file)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    try:
        generate_video(
            prompt_text,
            args.output,
            model_id=args.model,
            output_gcs_uri=args.gcs_uri,
            aspect_ratio=args.aspect_ratio,
            duration_seconds=args.duration,
            resolution=args.resolution,
            generate_audio=args.audio,
            person_generation=args.person_generation,
            seed=args.seed,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
