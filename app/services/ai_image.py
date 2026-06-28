"""
AI Image Generation Service

Provider abstraction for generating images from text prompts.
Supports: Pollinations (free), DALL-E, fal.ai (Flux), and extensible for more.

Usage:
    from app.services.ai_image import generate_image, generate_images_for_scenes
    image_path = generate_image("a cute cat in space", save_dir="/tmp/images")
"""

import base64
import os
import time
from typing import List, Optional
from urllib.parse import quote

import requests
from loguru import logger

from app.config import config
from app.utils import utils


def _get_provider() -> str:
    """Get configured AI image provider."""
    return config.app.get("ai_image_provider", "pollinations").strip().lower()


def _get_save_dir(task_id: str) -> str:
    """Get task-specific image save directory."""
    save_dir = os.path.join(utils.task_dir(task_id), "ai_images")
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def generate_image_pollinations(
    prompt: str,
    save_path: str,
    width: int = 1080,
    height: int = 1920,
    model: str = "flux",
    seed: Optional[int] = None,
) -> str:
    """
    Generate image using Pollinations AI (free, no API key).
    GET https://image.pollinations.ai/prompt/{prompt}?width=W&height=H&model=M
    """
    encoded_prompt = quote(prompt[:500])  # limit URL length
    params = f"width={width}&height={height}&model={model}&nologo=true"
    if seed is not None:
        params += f"&seed={seed}"
    url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?{params}"

    logger.info(f"pollinations image: {url[:120]}...")

    try:
        resp = requests.get(url, timeout=(30, 120))
        resp.raise_for_status()

        if len(resp.content) < 1000:
            logger.error(f"pollinations returned too small image: {len(resp.content)} bytes")
            return ""

        with open(save_path, "wb") as f:
            f.write(resp.content)

        logger.info(f"image saved: {save_path} ({len(resp.content)} bytes)")
        return save_path
    except Exception as e:
        logger.error(f"pollinations image generation failed: {e}")
        return ""


def generate_image_dalle(
    prompt: str,
    save_path: str,
    size: str = "1024x1792",  # DALL-E 3 portrait
    quality: str = "standard",
) -> str:
    """Generate image using OpenAI DALL-E 3."""
    api_key = config.app.get("openai_api_key", "")
    base_url = config.app.get("openai_base_url", "https://api.openai.com/v1")

    if not api_key:
        logger.error("OpenAI API key not configured for DALL-E")
        return ""

    try:
        resp = requests.post(
            f"{base_url}/images/generations",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "dall-e-3",
                "prompt": prompt[:4000],
                "n": 1,
                "size": size,
                "quality": quality,
                "response_format": "b64_json",
            },
            timeout=(30, 120),
        )
        resp.raise_for_status()
        data = resp.json()

        if "data" in data and len(data["data"]) > 0:
            image_data = data["data"][0].get("b64_json", "")
            if image_data:
                with open(save_path, "wb") as f:
                    f.write(base64.b64decode(image_data))
                logger.info(f"DALL-E image saved: {save_path}")
                return save_path
    except Exception as e:
        logger.error(f"DALL-E image generation failed: {e}")

    return ""


def generate_image_fal(
    prompt: str,
    save_path: str,
    width: int = 1080,
    height: int = 1920,
) -> str:
    """
    Generate image using fal.ai (has Flux models starting at $0.003/image).
    Uses fal-ai/flux/schnell by default (cheapest).
    """
    fal_key = config.app.get("fal_key", "").strip()
    if not fal_key:
        logger.warning("fal_key not configured, falling back to pollinations")
        return generate_image_pollinations(prompt, save_path, width, height)

    headers = {
        "Authorization": f"Key {fal_key}",
        "Content-Type": "application/json",
    }

    # Map dimensions to size
    aspect = width / height
    if abs(aspect - 9/16) < 0.1 or abs(aspect - 16/9) < 0.1:
        image_size = "portrait_16_9" if height > width else "landscape_16_9"
    else:
        image_size = "square_hd"

    payload = {
        "prompt": prompt[:1000],
        "image_size": image_size,
        "num_images": 1,
        "enable_safety_checker": False,
    }

    model = config.app.get("fal_image_model", "fal-ai/flux/schnell")

    try:
        logger.info(f"fal.ai image: {model}")
        resp = requests.post(
            f"https://queue.fal.run/{model}",
            headers=headers,
            json=payload,
            timeout=(30, 120),
        )
        resp.raise_for_status()
        data = resp.json()

        # fal.ai returns images in different formats
        images = data.get("images", [])
        if images:
            # {url: ..., content_type: ...}
            img_url = images[0].get("url", "")
            if not img_url:
                img_url = images[0] if isinstance(images[0], str) else ""
        else:
            # Maybe under {image: {url: ...}} or {output: {url: ...}}
            img_url = _extract_fal_image_url(data)

        if not img_url:
            logger.error(f"fal.ai: no image URL in response")
            return ""

        # Download the image
        img_resp = requests.get(img_url, timeout=(30, 120))
        img_resp.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(img_resp.content)
        logger.info(f"fal.ai image saved: {save_path}")
        return save_path

    except requests.exceptions.HTTPError as e:
        err_text = resp.text[:200] if hasattr(resp, "text") else str(e)
        if "balance" in err_text.lower() or "locked" in err_text.lower():
            logger.warning(f"fal.ai balance exhausted, falling back to pollinations")
            return generate_image_pollinations(prompt, save_path, width, height)
        logger.error(f"fal.ai image generation failed: {e} - {err_text}")
        return ""
    except Exception as e:
        logger.error(f"fal.ai image generation failed: {e}")
        return ""


def _extract_fal_image_url(data: dict) -> str:
    """Extract image URL from fal.ai response."""
    for key in ["image", "output", "result"]:
        item = data.get(key, {})
        if isinstance(item, dict):
            url = item.get("url", "")
            if url:
                return url
            nested = item.get("image", {}).get("url", "")
            if nested:
                return nested
    return ""


def generate_image(
    prompt: str,
    save_path: str,
    width: int = 1080,
    height: int = 1920,
    provider: Optional[str] = None,
) -> str:
    """
    Generate a single image using the configured provider.

    Args:
        prompt: Text description of the image
        save_path: Where to save the image file
        width: Image width in pixels
        height: Image height in pixels
        provider: Override provider ("pollinations", "dalle", "fal")

    Returns:
        Path to saved image, or empty string on failure
    """
    provider = provider or _get_provider()

    if provider == "pollinations":
        return generate_image_pollinations(prompt, save_path, width, height)
    elif provider == "dalle":
        # DALL-E sizes: 1024x1024, 1792x1024, 1024x1792
        size = "1024x1792" if height > width else "1792x1024"
        return generate_image_dalle(prompt, save_path, size=size)
    elif provider == "fal":
        return generate_image_fal(prompt, save_path, width, height)
    else:
        logger.error(f"unknown ai_image provider: {provider}")
        return ""


def generate_images_for_scenes(
    task_id: str,
    image_prompts: List[str],
    width: int = 1080,
    height: int = 1920,
    style_prompt: str = "",
) -> List[str]:
    """
    Generate one image per scene prompt.

    Args:
        task_id: Task identifier
        image_prompts: List of text prompts, one per scene
        width: Target image width
        height: Target image height
        style_prompt: Optional style prefix (e.g., "cartoon style, vibrant colors")

    Returns:
        List of image file paths
    """
    save_dir = _get_save_dir(task_id)
    image_paths = []

    for i, prompt in enumerate(image_prompts):
        full_prompt = f"{style_prompt} {prompt}".strip() if style_prompt else prompt
        save_path = os.path.join(save_dir, f"scene_{i:03d}.png")

        logger.info(f"generating image {i+1}/{len(image_prompts)}: {prompt[:80]}...")

        result = generate_image(full_prompt, save_path, width, height)
        if result:
            image_paths.append(result)
        else:
            logger.warning(f"failed to generate image for scene {i+1}, skipping")

        # Small delay to avoid rate limiting on free providers
        if i < len(image_prompts) - 1:
            time.sleep(1)

    logger.success(f"generated {len(image_paths)}/{len(image_prompts)} images")
    return image_paths
