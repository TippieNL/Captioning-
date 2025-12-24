from __future__ import annotations

import io
import os
import time
from typing import Any, Dict, Optional

import anyio
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import AutoFeatureExtractor, AutoTokenizer, VisionEncoderDecoderModel

from .prompts import PROMPTS, build_prompt

# TINY MODEL MODE (MOBILE SAFE)
# This replaces JoyCaption for Termux / CPU testing
USE_TINY_MODEL = True
MODEL_NAME = "nlpconnect/vit-gpt2-image-captioning"
MAX_FILE_SIZE = 20 * 1024 * 1024
REQUEST_TIMEOUT_S = 180

MODEL_CACHE: Dict[str, Any] = {
    "model": None,
    "tokenizer": None,
    "feature_extractor": None,
    "device": "cpu",
}
MODEL_LOCK = anyio.Lock()


app = FastAPI(title="JoyCaption Beta One")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_model() -> None:
    if not USE_TINY_MODEL:
        raise RuntimeError("Tiny model mode disabled.")
    if MODEL_NAME not in {
        "nlpconnect/vit-gpt2-image-captioning",
        "Salesforce/blip-image-captioning-base",
        "Salesforce/blip-image-captioning-small",
    }:
        raise RuntimeError("Unsupported tiny model.")

    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)

    model.to("cpu")
    model.eval()

    MODEL_CACHE["model"] = model
    MODEL_CACHE["tokenizer"] = tokenizer
    MODEL_CACHE["feature_extractor"] = feature_extractor
    MODEL_CACHE["device"] = "cpu"


def _get_model() -> Dict[str, Any]:
    if MODEL_CACHE["model"] is None or MODEL_CACHE["tokenizer"] is None:
        raise RuntimeError("Model not loaded")
    return MODEL_CACHE


def _clean_output(text: str) -> str:
    return " ".join(text.strip().split())


def _generate_caption(
    image: Image.Image,
    max_length: int,
    num_beams: int,
) -> str:
    cache = _get_model()
    model = cache["model"]
    tokenizer = cache["tokenizer"]
    feature_extractor = cache["feature_extractor"]

    inputs = feature_extractor(images=[image], return_tensors="pt")
    pixel_values = inputs["pixel_values"]
    output_ids = model.generate(
        pixel_values=pixel_values,
        max_length=max_length,
        num_beams=num_beams,
    )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return _clean_output(text)


@app.on_event("startup")
async def startup_event() -> None:
    if os.getenv("JOYCAPTION_LAZY_LOAD", "0") == "1":
        return
    async with MODEL_LOCK:
        if MODEL_CACHE["model"] is None:
            await anyio.to_thread.run_sync(_load_model)


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model": "vit-gpt2-image-captioning",
        "device": "cpu",
        "tiny_mode": True,
    }


@app.post("/api/caption")
async def caption_image(
    image: UploadFile = File(...),
    mode: str = Form(...),
    tone: Optional[str] = Form(None),
    length: Optional[str] = Form(None),
    word_count: Optional[int] = Form(None),
    seed: Optional[int] = Form(None),
    temperature: Optional[float] = Form(None),
    top_p: Optional[float] = Form(None),
    max_new_tokens: Optional[int] = Form(None),
) -> Dict[str, Any]:
    if mode not in PROMPTS:
        raise HTTPException(status_code=400, detail="Unsupported mode")
    if tone and tone not in {"formal", "casual", "neutral"}:
        raise HTTPException(status_code=400, detail="Invalid tone")
    if length and length not in {"short", "medium", "long"}:
        raise HTTPException(status_code=400, detail="Invalid length")
    if word_count is not None and (word_count <= 0 or word_count > 500):
        raise HTTPException(status_code=400, detail="Invalid word_count")

    data = await image.read()
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")

    try:
        pil_image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Invalid image") from exc

    prompt = build_prompt(mode=mode, tone=tone, length=length, word_count=word_count)

    start = time.perf_counter()
    try:
        async with MODEL_LOCK:
            if MODEL_CACHE["model"] is None:
                await anyio.to_thread.run_sync(_load_model)

        with anyio.fail_after(REQUEST_TIMEOUT_S):
            caption = await anyio.to_thread.run_sync(
                _generate_caption,
                pil_image,
                64,
                4,
            )
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise HTTPException(status_code=500, detail="Out of memory") from exc
        raise HTTPException(status_code=500, detail="Model error") from exc

    timing_ms = int((time.perf_counter() - start) * 1000)

    return {
        "prompt_used": prompt,
        "caption": caption,
        "mode": mode,
        "settings": {
            "tone": tone,
            "length": length,
            "word_count": word_count,
            "seed": seed,
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
        },
        "timing_ms": timing_ms,
    }
