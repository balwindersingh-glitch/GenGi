"""
Web UI backend for Veo and Nano Banana Pro video generation.
Run: uvicorn app:app --reload
"""
import warnings

# Suppress known warnings when using Python 3.9 / system LibreSSL
warnings.filterwarnings("ignore", message=".*Python version 3.9 past its end of life.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL.*")

import base64
import json
import os
import subprocess
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

# Vercel / serverless: if key is in env as JSON string, write to temp file so GCP libs can use it
_creds_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
if _creds_json and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
    try:
        _f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        _f.write(_creds_json)
        _f.close()
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = _f.name
    except Exception:
        pass

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from generate_video import generate_video as veo_generate_video, extend_video as veo_extend_video, get_config
from generate_image import generate_image as nano_banana_generate_image
from nano_banana import generate_video as nano_banana_generate_video
from analyze_video import analyze_video_for_prompts

app = FastAPI(title="Video Generation (Veo + Nano Banana Pro)")
executor = ThreadPoolExecutor(max_workers=2)

OUTPUT_DIR = Path(os.environ.get("VEO_OUTPUT_DIR", tempfile.gettempdir())) / "veo_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_OUTPUT_DIR = OUTPUT_DIR / "images"
IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HISTORY_PATH = Path(__file__).parent / "history.json"
jobs: dict = {}


def load_history() -> list:
    if not HISTORY_PATH.is_file():
        return []
    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_history(history: list) -> None:
    try:
        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history[-500:], f, indent=2)
    except Exception:
        pass


class ImagePart(BaseModel):
    data: str  # base64
    mime_type: str = "image/png"


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    provider: str = "veo"
    aspect_ratio: str = "16:9"
    duration_seconds: int = 6
    resolution: str = "720p"
    generate_audio: bool = False
    model_id: Optional[str] = None
    seed: Optional[int] = None
    person_generation: str = "allow_adult"
    negative_prompt: Optional[str] = None
    upload_drive: bool = True
    reference_images: Optional[List[ImagePart]] = None  # max 3 for Veo
    first_frame: Optional[ImagePart] = None
    last_frame: Optional[ImagePart] = None


class GenerateImageRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    aspect_ratio: str = "16:9"
    resolution: str = "1K"
    negative_prompt: Optional[str] = None
    model_id: Optional[str] = None  # e.g. gemini-2.5-flash-image or gemini-3-pro-image-preview
    images: Optional[List[ImagePart]] = None  # up to 3 reference images (base64)
    upload_drive: bool = True


class ExtendVideoRequest(BaseModel):
    job_id: str = Field(..., min_length=1)
    prompt: str = Field(..., min_length=1, description="What happens in the next 7 seconds")
    duration_seconds: int = Field(7, description="Total extension: 7 or 14 (15s display). 14 = two 7s extends.")
    prompt_2: Optional[str] = Field(None, description="Prompt for second 7s when duration_seconds=14")


def _mux_audio_into_extended_video(
    video_15s_path: Path,
    audio_source_8s_path: Path,
    output_path: Path,
    total_duration_sec: int = 15,
) -> bool:
    """First 0–8s get sound from the same Veo generate call (generate_audio=True → one API returns video+audio).
    The next 7s come from a different API (extend), which only returns video—no way to 'generate audio for this 7s'
    the same way. So we mux 15s video with 8s audio + last 7s of same audio. Returns True on success."""
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(audio_source_8s_path)],
            capture_output=True,
            timeout=10,
        )
        if probe.returncode != 0 or b"audio" not in (probe.stdout or b""):
            return False
        muxed = video_15s_path.parent / f"{video_15s_path.stem}_muxed.mp4"
        # 0–8s: original audio; 8–15s: last 7s of same audio, cross-faded at boundary (0.5s overlap), then pad to 15s
        filter_cplx = (
            "[1:a]atrim=0:8,asetpts=PTS-STARTPTS[a8];"
            "[1:a]atrim=1:8,asetpts=PTS-STARTPTS[a7];"
            f"[a8][a7]acrossfade=d=0.5:c1=tri:c2=tri,apad=whole_dur={total_duration_sec}[a]"
        )
        cmd = [
            "ffmpeg", "-y", "-nostdin",
            "-i", str(video_15s_path),
            "-i", str(audio_source_8s_path),
            "-filter_complex", filter_cplx,
            "-map", "0:v", "-map", "[a]",
            "-c:v", "copy", "-c:a", "aac", "-shortest",
            str(muxed),
        ]
        r = subprocess.run(cmd, capture_output=True, timeout=120)
        if r.returncode != 0 or not muxed.is_file():
            return False
        audio_source_8s_path.unlink(missing_ok=True)
        video_15s_path.unlink(missing_ok=True)
        muxed.replace(output_path)
        return True
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def upload_to_drive_safe(local_path: str, name: str, mime_type: Optional[str] = None) -> Optional[dict]:
    try:
        from drive_upload import upload_to_drive
        return upload_to_drive(local_path, name=name, mime_type=mime_type)
    except Exception:
        return None


def run_generation(job_id: str, req: GenerateRequest) -> None:
    output_path = OUTPUT_DIR / f"{job_id}.mp4"
    prompt_preview = (req.prompt[:60] + "…") if len(req.prompt) > 60 else req.prompt
    record = {
        "job_id": job_id,
        "prompt": req.prompt,
        "provider": req.provider,
        "status": "generating",
        "drive_url": None,
        "drive_id": None,
        "created_at": time.time(),
    }
    history = load_history()
    history.append(record)
    save_history(history)

    try:
        jobs[job_id]["status"] = "generating"
        if req.provider == "nano_banana":
            nano_banana_generate_video(
                req.prompt,
                str(output_path),
                duration_seconds=req.duration_seconds,
                resolution=req.resolution,
                aspect_ratio=req.aspect_ratio,
            )
        else:
            reference_images = None
            if req.reference_images and len(req.reference_images) > 0:
                reference_images = [
                    (base64.b64decode(img.data), img.mime_type or "image/png")
                    for img in req.reference_images[:3]
                ]
            first_frame = None
            if req.first_frame:
                first_frame = (base64.b64decode(req.first_frame.data), req.first_frame.mime_type or "image/png")
            last_frame = None
            if req.last_frame:
                last_frame = (base64.b64decode(req.last_frame.data), req.last_frame.mime_type or "image/png")
            initial_duration = 8 if req.duration_seconds == 15 else req.duration_seconds
            veo_generate_video(
                req.prompt,
                str(output_path),
                aspect_ratio=req.aspect_ratio,
                duration_seconds=initial_duration,
                resolution=req.resolution,
                generate_audio=req.generate_audio,
                model_id=req.model_id or get_config()["model_id"],
                seed=req.seed,
                person_generation=req.person_generation,
                negative_prompt=req.negative_prompt or None,
                reference_images=reference_images,
                first_frame=first_frame,
                last_frame=last_frame,
            )
            if req.duration_seconds == 15:
                cfg = get_config()
                gcs_uri = (cfg.get("output_gcs_uri") or os.environ.get("VEO_OUTPUT_GCS_URI") or "").strip().rstrip("/")
                if not gcs_uri:
                    raise ValueError("15s duration requires VEO_OUTPUT_GCS_URI for extend.")
                extended_path = OUTPUT_DIR / f"{job_id}_extended.mp4"
                veo_extend_video(
                    str(output_path),
                    "The scene continues naturally with the same cinematic style, motion and lighting.",
                    str(extended_path),
                    aspect_ratio="16:9",
                    output_gcs_uri=f"{gcs_uri}/extend_{job_id}",
                    generate_audio=req.generate_audio,
                )
                if req.generate_audio and output_path.is_file():
                    audio_src = OUTPUT_DIR / f"{job_id}_8s_audio.mp4"
                    output_path.rename(audio_src)
                    extended_path.rename(output_path)
                    if not _mux_audio_into_extended_video(output_path, audio_src, output_path, total_duration_sec=15):
                        audio_src.unlink(missing_ok=True)
                else:
                    extended_path.replace(output_path)

        jobs[job_id]["status"] = "done"
        jobs[job_id]["output_path"] = str(output_path)
        jobs[job_id]["kind"] = "video"
        drive_info = None
        if req.upload_drive and output_path.is_file():
            drive_info = upload_to_drive_safe(str(output_path), f"veo_{job_id}.mp4", mime_type="video/mp4")
            if drive_info:
                jobs[job_id]["drive_url"] = drive_info.get("webViewLink") or drive_info.get("webContentLink")
                jobs[job_id]["drive_id"] = drive_info.get("id")

        hist = load_history()
        for r in hist:
            if r.get("job_id") == job_id:
                r["status"] = "done"
                r["drive_url"] = (drive_info or {}).get("webViewLink") or (drive_info or {}).get("webContentLink")
                r["drive_id"] = (drive_info or {}).get("id")
                break
        save_history(hist)
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        hist = load_history()
        for r in hist:
            if r.get("job_id") == job_id:
                r["status"] = "failed"
                r["error"] = str(e)
                break
        save_history(hist)


def run_extend(job_id: str, extend_job_id: str, prompt: str, upload_drive: bool = True) -> None:
    """Extend the video from job_id by 7s; save result as extend_job_id."""
    if job_id not in jobs or jobs[job_id].get("status") != "done" or not jobs[job_id].get("output_path"):
        jobs[extend_job_id]["status"] = "failed"
        jobs[extend_job_id]["error"] = "Source job not found or not ready."
        return
    input_path = Path(jobs[job_id]["output_path"])
    if not input_path.is_file():
        jobs[extend_job_id]["status"] = "failed"
        jobs[extend_job_id]["error"] = "Source video file missing."
        return
    output_path = OUTPUT_DIR / f"{extend_job_id}.mp4"
    record = {
        "job_id": extend_job_id,
        "prompt": prompt,
        "provider": "veo",
        "status": "generating",
        "drive_url": None,
        "drive_id": None,
        "created_at": time.time(),
        "extended_from": job_id,
    }
    history = load_history()
    history.append(record)
    save_history(history)
    try:
        jobs[extend_job_id]["status"] = "generating"
        jobs[extend_job_id]["extended_from"] = job_id
        cfg = get_config()
        gcs_uri = (cfg.get("output_gcs_uri") or os.environ.get("VEO_OUTPUT_GCS_URI") or "").strip().rstrip("/")
        if not gcs_uri:
            raise ValueError(
                "Extend requires VEO_OUTPUT_GCS_URI. Set it to a GCS path (e.g. gs://your-bucket/veo-output/)."
            )
        veo_extend_video(
            str(input_path),
            prompt,
            str(output_path),
            aspect_ratio="16:9",
            output_gcs_uri=f"{gcs_uri}/extend_{extend_job_id}",
        )
        jobs[extend_job_id]["status"] = "done"
        jobs[extend_job_id]["output_path"] = str(output_path)
        jobs[extend_job_id]["kind"] = "video"
        drive_info = None
        if upload_drive and output_path.is_file():
            drive_info = upload_to_drive_safe(str(output_path), f"veo_ext_{extend_job_id}.mp4", mime_type="video/mp4")
            if drive_info:
                jobs[extend_job_id]["drive_url"] = drive_info.get("webViewLink") or drive_info.get("webContentLink")
                jobs[extend_job_id]["drive_id"] = drive_info.get("id")
        hist = load_history()
        for r in hist:
            if r.get("job_id") == extend_job_id:
                r["status"] = "done"
                r["drive_url"] = (drive_info or {}).get("webViewLink") or (drive_info or {}).get("webContentLink")
                r["drive_id"] = (drive_info or {}).get("id")
                break
        save_history(hist)
    except Exception as e:
        jobs[extend_job_id]["status"] = "failed"
        jobs[extend_job_id]["error"] = str(e)
        hist = load_history()
        for r in hist:
            if r.get("job_id") == extend_job_id:
                r["status"] = "failed"
                r["error"] = str(e)
                break
        save_history(hist)


def run_image_generation(job_id: str, req: GenerateImageRequest) -> None:
    output_path = IMAGE_OUTPUT_DIR / f"{job_id}.png"
    record = {
        "job_id": job_id,
        "prompt": req.prompt,
        "provider": "nano_banana_image",
        "status": "generating",
        "drive_url": None,
        "drive_id": None,
        "created_at": time.time(),
    }
    history = load_history()
    history.append(record)
    save_history(history)
    try:
        jobs[job_id]["status"] = "generating"
        reference_images = None
        if req.images and len(req.images) > 0:
            import base64
            reference_images = []
            for img in req.images[:3]:
                raw = base64.b64decode(img.data)
                reference_images.append((raw, img.mime_type or "image/png"))
        nano_banana_generate_image(
            req.prompt,
            str(output_path),
            aspect_ratio=req.aspect_ratio,
            resolution=req.resolution,
            negative_prompt=req.negative_prompt,
            model_id=req.model_id,
            reference_images=reference_images,
        )
        jobs[job_id]["status"] = "done"
        jobs[job_id]["output_path"] = str(output_path)
        jobs[job_id]["kind"] = "image"
        drive_info = None
        if req.upload_drive and output_path.is_file():
            drive_info = upload_to_drive_safe(str(output_path), f"nano_banana_{job_id}.png", mime_type="image/png")
            if drive_info:
                jobs[job_id]["drive_url"] = drive_info.get("webViewLink") or drive_info.get("webContentLink")
        hist = load_history()
        for r in hist:
            if r.get("job_id") == job_id:
                r["status"] = "done"
                r["drive_url"] = (drive_info or {}).get("webViewLink") or (drive_info or {}).get("webContentLink")
                break
        save_history(hist)
    except Exception as e:
        err_msg = str(e)
        if "404" in err_msg and ("model" in err_msg.lower() or "not found" in err_msg.lower()):
            err_msg = "Model not available in your project. Try 'Nano Banana' instead."
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = err_msg
        hist = load_history()
        for r in hist:
            if r.get("job_id") == job_id:
                r["status"] = "failed"
                r["error"] = err_msg
                break
        save_history(hist)


@app.post("/api/generate")
def api_generate(req: GenerateRequest):
    if req.provider == "veo":
        if not get_config()["project_id"]:
            raise HTTPException(
                status_code=503,
                detail="Set GOOGLE_CLOUD_PROJECT or GCP_PROJECT_ID for Veo.",
            )
        if req.reference_images and len(req.reference_images) > 3:
            raise HTTPException(status_code=400, detail="Maximum 3 reference images allowed for Veo.")
    if req.provider == "nano_banana" and not os.environ.get("NANO_BANANA_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="Set NANO_BANANA_API_KEY for Nano Banana Pro.",
        )
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "output_path": None, "error": None, "drive_url": None, "kind": "video"}
    executor.submit(run_generation, job_id, req)
    return {"job_id": job_id}


@app.post("/api/extend-video")
def api_extend_video(req: ExtendVideoRequest):
    """Extend a completed Veo video by 7 seconds. Uses veo-3.1-generate-preview. Requires VEO_OUTPUT_GCS_URI."""
    cfg = get_config()
    if not cfg["project_id"]:
        raise HTTPException(status_code=503, detail="Set GOOGLE_CLOUD_PROJECT or GCP credentials for Veo.")
    gcs_uri = (cfg.get("output_gcs_uri") or os.environ.get("VEO_OUTPUT_GCS_URI") or "").strip().rstrip("/")
    if not gcs_uri:
        raise HTTPException(
            status_code=400,
            detail="Extend requires VEO_OUTPUT_GCS_URI (e.g. gs://your-bucket/veo-output/). Set it in your environment.",
        )
    if req.job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found.")
    if jobs[req.job_id].get("status") != "done" or jobs[req.job_id].get("kind") != "video":
        raise HTTPException(status_code=400, detail="Job must be a completed Veo video.")
    duration = 14 if req.duration_seconds == 14 else 7
    if duration == 14:
        extend_job_id_1 = str(uuid.uuid4())
        extend_job_id_2 = str(uuid.uuid4())
        jobs[extend_job_id_1] = {"status": "pending", "output_path": None, "error": None, "drive_url": None, "kind": "video"}
        jobs[extend_job_id_2] = {"status": "generating", "output_path": None, "error": None, "drive_url": None, "kind": "video"}
        second_prompt = (req.prompt_2 or "").strip() or req.prompt

        def run_double_extend():
            run_extend(req.job_id, extend_job_id_1, req.prompt, upload_drive=False)
            run_extend(extend_job_id_1, extend_job_id_2, second_prompt, upload_drive=True)

        executor.submit(run_double_extend)
        return {"job_id": extend_job_id_2}
    extend_job_id = str(uuid.uuid4())
    jobs[extend_job_id] = {"status": "pending", "output_path": None, "error": None, "drive_url": None, "kind": "video"}
    executor.submit(run_extend, req.job_id, extend_job_id, req.prompt, upload_drive=True)
    return {"job_id": extend_job_id}


MAX_ANALYZE_VIDEO_MB = 50


@app.post("/api/analyze-video")
def api_analyze_video(video: UploadFile):
    """Upload a video; returns per-segment descriptions and replication prompts for Veo + character."""
    if not get_config()["project_id"]:
        raise HTTPException(
            status_code=503,
            detail="Set GOOGLE_CLOUD_PROJECT or GCP credentials for video analysis.",
        )
    if not video.content_type or not video.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Please upload a video file (e.g. MP4).")
    try:
        data = video.file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read upload: {e}") from e
    if len(data) > MAX_ANALYZE_VIDEO_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"Video too large. Max {MAX_ANALYZE_VIDEO_MB} MB for analysis.",
        )
    suffix = Path(video.filename or "video.mp4").suffix or ".mp4"
    if suffix.lower() not in (".mp4", ".webm", ".mov", ".quicktime"):
        suffix = ".mp4"
    path = OUTPUT_DIR / f"_analyze_{uuid.uuid4().hex}{suffix}"
    try:
        path.write_bytes(data)
        segments = analyze_video_for_prompts(str(path))
        return {"segments": segments}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    finally:
        if path.is_file():
            try:
                path.unlink()
            except Exception:
                pass


@app.post("/api/generate-image")
def api_generate_image(req: GenerateImageRequest):
    if not get_config()["project_id"]:
        raise HTTPException(status_code=503, detail="Same as Veo: set GOOGLE_CLOUD_PROJECT or use your GCP credentials.")
    if req.images and len(req.images) > 3:
        raise HTTPException(status_code=400, detail="Maximum 3 reference images allowed.")
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "output_path": None, "error": None, "drive_url": None, "kind": "image"}
    executor.submit(run_image_generation, job_id, req)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def api_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    j = jobs[job_id]
    out = {"job_id": job_id, "status": j["status"]}
    if j.get("error"):
        out["error"] = j["error"]
    if j.get("output_path"):
        if j.get("kind") == "image":
            out["image_url"] = f"/api/jobs/{job_id}/image"
        else:
            out["video_url"] = f"/api/jobs/{job_id}/video"
    if j.get("drive_url"):
        out["drive_url"] = j["drive_url"]
    if j.get("extended_from"):
        out["extended_from"] = j["extended_from"]
    return out


@app.get("/api/jobs/{job_id}/video")
def api_job_video(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    j = jobs[job_id]
    if j["status"] != "done" or not j.get("output_path"):
        raise HTTPException(status_code=404, detail="Video not ready")
    path = Path(j["output_path"])
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Video file missing")
    return FileResponse(path, media_type="video/mp4", filename=f"video_{job_id}.mp4")


@app.get("/api/jobs/{job_id}/image")
def api_job_image(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    j = jobs[job_id]
    if j["status"] != "done" or not j.get("output_path"):
        raise HTTPException(status_code=404, detail="Image not ready")
    path = Path(j["output_path"])
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Image file missing")
    return FileResponse(
        path,
        media_type="image/png",
        filename=f"image_{job_id}.png",
        content_disposition_type="inline",
    )


@app.get("/api/history")
def api_history():
    return load_history()


@app.get("/api/config")
def api_config():
    cfg = get_config()
    return {
        "project_configured": bool(cfg["project_id"]),
        "location": cfg["location"],
        "model_id": cfg["model_id"],
        "nano_banana_image_uses_same_key": True,
        "nano_banana_configured": bool(os.environ.get("NANO_BANANA_API_KEY")),
        "drive_configured": bool(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")),
    }


FRONTEND = Path(__file__).parent / "frontend"
INDEX_HTML = FRONTEND / "index.html"


@app.get("/")
def index():
    if INDEX_HTML.is_file():
        return FileResponse(INDEX_HTML)
    return {"message": "Create frontend/index.html for the UI."}
