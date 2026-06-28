"""
AI Video Generation Service

Provider abstraction for generating videos from images (image-to-video) or text.
Supports: Kling (direct), fal.ai (Kling proxy), and extensible for Runway, Pika, Luma, etc.

Usage:
    from app.services.ai_video import generate_video_from_image, generate_videos_for_scenes
    video_path = generate_video_from_image(image_path, prompt, save_dir="/tmp/videos")

Kling API flow (simplified Bearer auth):
    1. Auth: Bearer <api-key> (from kling.ai/dev/api-key)
    2. POST /v1/videos/image2video → returns task_id
    3. Poll GET /v1/videos/image2video/{task_id} → wait for completion
    4. Download result video
"""

import os
import time
import requests
from typing import List, Optional
from loguru import logger

import json

from app.config import config
from app.utils import utils


# ─── Helpers ───────────────────────────────────────────────────────────

def _get_provider() -> str:
    """Get configured AI video provider."""
    return config.app.get("ai_video_provider", "kling").strip().lower()


def _get_kling_base_url() -> str:
    """Get Kling API base URL."""
    return config.app.get("kling_base_url", "https://api.klingai.com").rstrip("/")


def _get_kling_api_key() -> str:
    """Get Kling API key from config."""
    api_key = config.app.get("kling_api_key", "").strip()
    if not api_key:
        # Fallback to access_key/secret_key for backward compat
        access_key = config.app.get("kling_access_key", "").strip()
        if access_key:
            return access_key
        raise ValueError(
            "Kling API key not configured. Set kling_api_key in config.toml "
            "(get it at https://kling.ai/dev/api-key)"
        )
    return api_key


def _get_kling_headers() -> dict:
    """Build auth header for Kling API (simple Bearer token)."""
    api_key = _get_kling_api_key()
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _get_save_dir(task_id: str) -> str:
    """Get task-specific video save directory."""
    save_dir = os.path.join(utils.task_dir(task_id), "ai_videos")
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def _image_to_base64(image_path: str) -> str:
    """Convert image file to base64 data URI."""
    import base64 as b64

    with open(image_path, "rb") as f:
        data = b64.b64encode(f.read()).decode("utf-8")

    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    mime = f"image/{ext}" if ext else "image/png"
    return f"data:{mime};base64,{data}"


def _download_video(url: str, save_path: str) -> str:
    """Download video from URL to local path."""
    try:
        resp = requests.get(url, timeout=(30, 300), stream=True)
        resp.raise_for_status()

        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"video downloaded: {save_path}")
        return save_path
    except Exception as e:
        logger.error(f"failed to download video: {e}")
        return ""


# ─── Kling Provider ────────────────────────────────────────────────────

def kling_image_to_video(
    image_path: str,
    prompt: str,
    save_path: str,
    duration: int = 5,
    mode: str = "std",
    aspect_ratio: str = "9:16",
) -> str:
    """
    Generate video from image using Kling API.

    Args:
        image_path: Path to source image
        prompt: Motion/camera description
        save_path: Where to save the resulting video
        duration: Video length in seconds (5 or 10)
        mode: Quality mode ("std" or "pro")
        aspect_ratio: "9:16", "16:9", or "1:1"

    Returns:
        Path to saved video, or empty string on failure
    """
    base_url = _get_kling_base_url()
    headers = _get_kling_headers()

    # Convert image to base64
    image_b64 = _image_to_base64(image_path)

    # Create task
    payload = {
        "model_name": config.app.get("kling_model_name", "kling-v3"),
        "image": image_b64,
        "prompt": prompt[:500],
        "negative_prompt": "",
        "duration": duration,
        "mode": mode,
    }

    try:
        logger.info(f"kling image2video: prompt='{prompt[:60]}...', duration={duration}s, mode={mode}")
        resp = requests.post(
            f"{base_url}/v1/videos/image2video",
            headers=headers,
            json=payload,
            timeout=(30, 60),
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("code") != 0:
            logger.error(f"kling API error: code={data.get('code')}, message={data.get('message', 'unknown')}")
            return ""

        task_id = data["data"]["task_id"]
        logger.info(f"kling task created: {task_id}")

        # Poll for completion
        video_url = _kling_poll_task(task_id, headers, base_url, "image2video")
        if not video_url:
            return ""

        # Download
        return _download_video(video_url, save_path)

    except Exception as e:
        logger.error(f"kling image2video failed: {e}")
        return ""


def kling_text_to_video(
    prompt: str,
    save_path: str,
    duration: int = 5,
    mode: str = "std",
    aspect_ratio: str = "9:16",
) -> str:
    """
    Generate video from text prompt using Kling API.
    """
    base_url = _get_kling_base_url()
    headers = _get_kling_headers()

    payload = {
        "model_name": config.app.get("kling_model_name", "kling-v3"),
        "prompt": prompt[:500],
        "negative_prompt": "",
        "duration": duration,
        "mode": mode,
        "aspect_ratio": aspect_ratio,
    }

    try:
        logger.info(f"kling text2video: prompt='{prompt[:60]}...'")
        resp = requests.post(
            f"{base_url}/v1/videos/text2video",
            headers=headers,
            json=payload,
            timeout=(30, 60),
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("code") != 0:
            logger.error(f"kling API error: code={data.get('code')}, message={data.get('message')}")
            return ""

        task_id = data["data"]["task_id"]
        logger.info(f"kling task created: {task_id}")

        video_url = _kling_poll_task(task_id, headers, base_url, "text2video")
        if not video_url:
            return ""

        return _download_video(video_url, save_path)

    except Exception as e:
        logger.error(f"kling text2video failed: {e}")
        return ""


def _kling_poll_task(
    task_id: str,
    headers: dict,
    base_url: str,
    endpoint: str = "image2video",
    max_wait: int = 600,
    interval: int = 10,
) -> str:
    """
    Poll Kling task until completion.

    Args:
        task_id: Kling task ID
        headers: Auth headers
        base_url: API base URL
        endpoint: "image2video" or "text2video"
        max_wait: Maximum wait time in seconds
        interval: Poll interval in seconds

    Returns:
        Video URL on success, empty string on failure/timeout
    """
    elapsed = 0
    while elapsed < max_wait:
        try:
            resp = requests.get(
                f"{base_url}/v1/videos/{endpoint}/{task_id}",
                headers=headers,
                timeout=(30, 60),
            )
            resp.raise_for_status()
            data = resp.json()

            if data.get("code") != 0:
                logger.error(f"kling query error: code={data.get('code')}, message={data.get('message')}")
                return ""

            task_data = data.get("data", {})
            status = task_data.get("task_status", "")

            logger.info(f"kling task {task_id}: status={status} ({elapsed}s elapsed)")

            if status == "succeed":
                videos = task_data.get("task_result", {}).get("videos", [])
                if videos:
                    video_url = videos[0].get("url", "")
                    if video_url:
                        return video_url
                logger.error("kling task succeeded but no video URL")
                return ""

            if status == "failed":
                logger.error(f"kling task failed: {task_data.get('task_status_msg', '')}")
                return ""

            # still processing (submitted, processing)
            time.sleep(interval)
            elapsed += interval

        except Exception as e:
            logger.warning(f"kling poll error (will retry): {e}")
            time.sleep(interval)
            elapsed += interval

    logger.error(f"kling task timed out after {max_wait}s")
    return ""


# ─── fal.ai Provider ───────────────────────────────────────────────────

def _get_fal_key() -> str:
    """Get fal.ai API key."""
    key = config.app.get("fal_key", "").strip()
    if not key:
        raise ValueError(
            "fal.ai key not configured. Set fal_key in config.toml "
            "(get it at https://fal.ai/dashboard/keys)"
        )
    return key


def _fal_headers() -> dict:
    return {"Authorization": f"Key {_get_fal_key()}", "Content-Type": "application/json"}


def _fal_submit_and_poll(
    endpoint: str,
    payload: dict,
    save_path: str,
    max_wait: int = 600,
    poll_interval: int = 5,
) -> str:
    """
    Submit request to fal.ai queue and poll for result.

    fal.ai pattern:
      1. POST to queue.fal.run/{endpoint} → get request_id
      2. Poll queue.fal.run/requests/{request_id}/status → status
      3. When completed, GET response URL for video

    Returns:
        Path to downloaded video, or "" on failure
    """
    base_url = "https://queue.fal.run"
    headers = _fal_headers()

    try:
        # Step 1: Submit
        logger.info(f"fal.ai submit: {endpoint}")
        resp = requests.post(
            f"{base_url}/{endpoint}",
            headers=headers,
            json=payload,
            timeout=(30, 60),
        )
        if resp.status_code != 200 and resp.status_code != 201:
            detail = resp.json().get("detail", "unknown")
            logger.error(f"fal.ai submit failed: {resp.status_code} {detail}")
            return ""

        data = resp.json()
        request_id = data.get("request_id")
        if not request_id:
            logger.error(f"fal.ai submit: no request_id in response")
            return ""

        logger.info(f"fal.ai task created: {request_id}")

        # Step 2: Poll
        elapsed = 0
        while elapsed < max_wait:
            try:
                status_resp = requests.get(
                    f"{base_url}/requests/{request_id}/status",
                    headers=headers,
                    timeout=30,
                )
                if status_resp.status_code != 200:
                    time.sleep(poll_interval)
                    elapsed += poll_interval
                    continue

                status_data = status_resp.json()
                status = status_data.get("status", "")

                logger.info(f"fal.ai {request_id}: {status} ({elapsed}s)")

                if status == "COMPLETED":
                    # Get the response
                    resp_get = requests.get(
                        f"{base_url}/requests/{request_id}",
                        headers=headers,
                        timeout=30,
                    )
                    if resp_get.status_code != 200:
                        logger.error("fal.ai: completed but failed to get response")
                        return ""

                    result = resp_get.json()
                    # fal.ai wraps results based on endpoint
                    video_url = _extract_fal_video_url(result, endpoint)
                    if video_url:
                        return _download_video(video_url, save_path)
                    logger.error(f"fal.ai: completed but no video URL in response")
                    return ""

                elif status in ("FAILED", "ERROR"):
                    err = status_data.get("error", "unknown")
                    logger.error(f"fal.ai task failed: {err}")
                    return ""

                # Still processing
                time.sleep(poll_interval)
                elapsed += poll_interval

            except Exception as e:
                logger.warning(f"fal.ai poll error: {e}")
                time.sleep(poll_interval)
                elapsed += poll_interval

        logger.error(f"fal.ai task timed out after {max_wait}s")
        return ""

    except Exception as e:
        logger.error(f"fal.ai request failed: {e}")
        return ""


def _extract_fal_video_url(result: dict, endpoint: str) -> str:
    """Extract video URL from fal.ai response, handles different endpoint formats."""
    # Many fal.ai endpoints return {video: {url: ...}} or {video: {url: {url: ...}}}
    if endpoint.startswith("fal-ai/kling"):
        video = result.get("video", {})
        if isinstance(video, dict):
            url = video.get("url", "")
            if isinstance(url, dict):
                url = url.get("url", "")
            return url
        return ""
    # General fallback
    for key in ["video", "output", "result"]:
        item = result.get(key, {})
        if isinstance(item, dict):
            url = item.get("url", "")
            if url:
                return url
    return ""


def fal_image_to_video(
    image_path: str,
    prompt: str,
    save_path: str,
    duration: int = 5,
    model: str = "kling-v2",
) -> str:
    """
    Generate video using fal.ai.

    Uses Kling model on fal.ai platform.
    Available Kling models on fal.ai:
      - fal-ai/kling-video/v1.6/standard/image-to-video
      - fal-ai/kling-video/v1.6/pro/image-to-video
    """
    import base64 as b64

    # Upload image to a hosted URL first, or use base64 via fal's built-in support
    with open(image_path, "rb") as f:
        img_data = b64.b64encode(f.read()).decode("utf-8")

    ext = os.path.splitext(image_path)[1].lstrip(".")
    mime = f"image/{ext}" if ext else "image/png"
    data_uri = f"data:{mime};base64,{img_data}"

    # Map model name to fal.ai endpoint
    model_map = {
        "kling-std": "fal-ai/kling-video/v2.1/standard/image-to-video",
        "kling-pro": "fal-ai/kling-video/v2.1/pro/image-to-video",
        "kling-master": "fal-ai/kling-video/v2.1/master/image-to-video",
    }

    model_key = config.app.get("fal_kling_model", model)
    endpoint = model_map.get(model_key, model_map["kling-std"])

    payload = {
        "image_url": data_uri,
        "prompt": prompt[:500],
        "duration": str(duration),
    }

    return _fal_submit_and_poll(endpoint, payload, save_path)


# ─── Public API ────────────────────────────────────────────────────────

def generate_video_from_image(
    image_path: str,
    prompt: str,
    save_path: str,
    duration: int = 5,
    mode: str = "std",
    aspect_ratio: str = "9:16",
    provider: Optional[str] = None,
) -> str:
    """
    Generate video from a single image using configured provider.

    Args:
        image_path: Path to source image
        prompt: Motion/camera description for animation
        save_path: Output video path
        duration: Target duration (5 or 10 seconds for Kling)
        mode: Quality mode ("std" or "pro")
        aspect_ratio: "9:16", "16:9", "1:1"
        provider: Override provider ("kling", "fal", "runway", "pika")

    Returns:
        Path to saved video, or empty string on failure
    """
    provider = provider or _get_provider()

    if provider == "kling":
        return kling_image_to_video(image_path, prompt, save_path, duration, mode, aspect_ratio)
    elif provider == "fal":
        return fal_image_to_video(image_path, prompt, save_path, duration)
    elif provider == "runway":
        # TODO: implement Runway provider
        logger.error("Runway provider not yet implemented")
        return ""
    elif provider == "pika":
        # TODO: implement Pika provider
        logger.error("Pika provider not yet implemented")
        return ""
    else:
        logger.error(f"unknown ai_video provider: {provider}")
        return ""


def generate_videos_for_scenes(
    task_id: str,
    image_paths: List[str],
    motion_prompts: List[str],
    duration: int = 5,
    mode: str = "std",
    aspect_ratio: str = "9:16",
) -> List[str]:
    """
    Generate videos from images, one per scene.

    Args:
        task_id: Task identifier
        image_paths: List of image paths (one per scene)
        motion_prompts: List of motion/camera descriptions (one per scene)
        duration: Per-clip duration in seconds
        mode: Quality mode
        aspect_ratio: Video aspect ratio

    Returns:
        List of video file paths
    """
    save_dir = _get_save_dir(task_id)
    video_paths = []

    n_scenes = min(len(image_paths), len(motion_prompts))

    for i in range(n_scenes):
        save_path = os.path.join(save_dir, f"scene_{i:03d}.mp4")
        logger.info(f"generating video {i+1}/{n_scenes}: {motion_prompts[i][:60]}...")

        result = generate_video_from_image(
            image_path=image_paths[i],
            prompt=motion_prompts[i],
            save_path=save_path,
            duration=duration,
            mode=mode,
            aspect_ratio=aspect_ratio,
        )

        if result:
            video_paths.append(result)
        else:
            logger.warning(f"failed to generate video for scene {i+1}")

    logger.success(f"generated {len(video_paths)}/{n_scenes} videos")
    return video_paths


def generate_text_to_videos(
    task_id: str,
    video_prompts: List[str],
    duration: int = 5,
    mode: str = "std",
    aspect_ratio: str = "9:16",
    provider: Optional[str] = None,
) -> List[str]:
    """
    Generate videos directly from text prompts (no image step).

    Args:
        task_id: Task identifier
        video_prompts: List of scene descriptions
        duration: Per-clip duration
        mode: Quality mode
        aspect_ratio: Video aspect ratio
        provider: Override provider

    Returns:
        List of video file paths
    """
    save_dir = _get_save_dir(task_id)
    video_paths = []
    p = provider or _get_provider()

    for i, prompt in enumerate(video_prompts):
        save_path = os.path.join(save_dir, f"scene_{i:03d}.mp4")
        logger.info(f"generating text2video {i+1}/{len(video_prompts)}: {prompt[:60]}...")

        if p == "kling":
            result = kling_text_to_video(prompt, save_path, duration, mode, aspect_ratio)
        else:
            logger.error(f"provider {p} not implemented for text2video")
            result = ""

        if result:
            video_paths.append(result)
        else:
            logger.warning(f"failed to generate video for scene {i+1}")

    logger.success(f"generated {len(video_paths)}/{len(video_prompts)} videos")
    return video_paths
