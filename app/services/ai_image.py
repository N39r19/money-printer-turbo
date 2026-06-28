"""
AI Image Generation Service

Provider abstraction for generating images from text prompts.
Supports: Pollinations (free), DALL-E, Flux, and extensible for more.

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
        provider: Override provider ("pollinations", "dalle")

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
