from __future__ import annotations

import io
import os
import logging
import time
from typing import Any, Dict, Optional

import anyio
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

from .prompts import PROMPTS, build_prompt

MODEL_NAME = "fancyfeast/llama-joycaption-beta-one-hf-llava"
MAX_FILE_SIZE = 20 * 1024 * 1024
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_NEW_TOKENS = 512
REQUEST_TIMEOUT_S = 180

MODEL_CACHE: Dict[str, Any] = {"model": None, "processor": None, "device": None}
MODEL_LOCK = anyio.Lock()


app = FastAPI(title="JoyCaption Beta One")

logging.basicConfig(level=logging.INFO)

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
    if MODEL_NAME != "fancyfeast/llama-joycaption-beta-one-hf-llava":
        raise RuntimeError("Only JoyCaption Beta One is supported.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    if device == "cpu":
        logging.warning("CUDA not available. Falling back to CPU.")

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    MODEL_CACHE["model"] = model
    MODEL_CACHE["processor"] = processor
    MODEL_CACHE["device"] = device


def _get_model() -> Dict[str, Any]:
    if MODEL_CACHE["model"] is None or MODEL_CACHE["processor"] is None:
        raise RuntimeError("Model not loaded")
    return MODEL_CACHE


def _ensure_model_loaded() -> None:
    if MODEL_CACHE["model"] is None:
        _load_model()


def _clean_output(text: str) -> str:
    return " ".join(text.strip().split())


def _generate_caption(
    image: Image.Image,
    prompt: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    seed: Optional[int],
) -> str:
    cache = _get_model()
    model = cache["model"]
    processor = cache["processor"]
    device = cache["device"]

    if seed is not None:
        torch.manual_seed(seed)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image"},
            ],
        }
    ]

    prompt_text = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=[prompt_text], images=[image], return_tensors="pt")

    if device == "cuda":
        inputs = {
            k: v.to(device=device, dtype=torch.bfloat16 if v.dtype.is_floating_point else v.dtype)
            for k, v in inputs.items()
        }
    else:
        inputs = {k: v.to(device=device) for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[1]

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
    )

    generated = output_ids[:, input_len:]
    text = processor.batch_decode(generated, skip_special_tokens=True)[0]
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
        "model_loaded": MODEL_CACHE["model"] is not None,
        "device": MODEL_CACHE["device"] or ("cuda" if torch.cuda.is_available() else "cpu"),
    }


@app.post("/api/caption")
async def caption_image(
    image: UploadFile = File(...),
    mode: str = Form(...),
    tone: Optional[str] = Form(None),
    length: Optional[str] = Form(None),
    word_count: Optional[int] = Form(None),
    seed: Optional[int] = Form(None),
    temperature: float = Form(DEFAULT_TEMPERATURE),
    top_p: float = Form(DEFAULT_TOP_P),
    max_new_tokens: int = Form(DEFAULT_MAX_NEW_TOKENS),
) -> Dict[str, Any]:
    if mode not in PROMPTS:
        raise HTTPException(status_code=400, detail="Unsupported mode")
    if tone and tone not in {"formal", "casual", "neutral"}:
        raise HTTPException(status_code=400, detail="Invalid tone")
    if length and length not in {"short", "medium", "long"}:
        raise HTTPException(status_code=400, detail="Invalid length")
    if word_count is not None and (word_count <= 0 or word_count > 500):
        raise HTTPException(status_code=400, detail="Invalid word_count")
    if temperature <= 0 or temperature > 2:
        raise HTTPException(status_code=400, detail="Invalid temperature")
    if top_p <= 0 or top_p > 1:
        raise HTTPException(status_code=400, detail="Invalid top_p")
    if max_new_tokens <= 0 or max_new_tokens > 1024:
        raise HTTPException(status_code=400, detail="Invalid max_new_tokens")

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
                prompt,
                float(temperature),
                float(top_p),
                int(max_new_tokens),
                seed,
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


@app.post("/api/caption/batch")
async def caption_batch(
    images: list[UploadFile] = File(...),
    mode: str = Form(...),
    tone: Optional[str] = Form(None),
    length: Optional[str] = Form(None),
    word_count: Optional[int] = Form(None),
    temperature: float = Form(DEFAULT_TEMPERATURE),
    top_p: float = Form(DEFAULT_TOP_P),
    max_new_tokens: int = Form(DEFAULT_MAX_NEW_TOKENS),
) -> Dict[str, Any]:
    results = []
    for upload in images:
        data = await upload.read()
        if len(data) > MAX_FILE_SIZE:
            results.append({"filename": upload.filename, "error": "File too large"})
            continue
        try:
            pil_image = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            results.append({"filename": upload.filename, "error": "Invalid image"})
            continue
        try:
            prompt = build_prompt(mode=mode, tone=tone, length=length, word_count=word_count)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            async with MODEL_LOCK:
                if MODEL_CACHE["model"] is None:
                    await anyio.to_thread.run_sync(_load_model)
            with anyio.fail_after(REQUEST_TIMEOUT_S):
                caption = await anyio.to_thread.run_sync(
                    _generate_caption,
                    pil_image,
                    prompt,
                    float(temperature),
                    float(top_p),
                    int(max_new_tokens),
                    None,
                )
            results.append({"filename": upload.filename, "caption": caption, "prompt_used": prompt})
        except RuntimeError as exc:
            results.append({"filename": upload.filename, "error": str(exc)})

    return {"mode": mode, "results": results}
