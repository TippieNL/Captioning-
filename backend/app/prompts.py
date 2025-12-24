from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional


@dataclass(frozen=True)
class PromptSpec:
    key: str
    label: str
    builder: Callable[[Optional[str], Optional[str], Optional[int]], str]


def _long_detailed(_: Optional[str], __: Optional[str], ___: Optional[int]) -> str:
    return "Write a long detailed description for this image."


def _detailed_word_count(_: Optional[str], __: Optional[str], word_count: Optional[int]) -> str:
    if not word_count:
        raise ValueError("word_count is required for this mode")
    return f"Write a detailed description for this image in {word_count} words or less."


def _length_tone(length: Optional[str], tone: Optional[str], _: Optional[int]) -> str:
    length_value = length or "medium"
    tone_value = tone or "neutral"
    return f"Write a {length_value} descriptive caption for this image in a {tone_value} tone."


def _straightforward(_: Optional[str], __: Optional[str], ___: Optional[int]) -> str:
    return (
        "Write a concise, objective caption describing only what is visible in the image. "
        "Preserve any exact visible text verbatim. Mention watermarks if present. "
        "Avoid subjective language, speculation, and figurative phrasing. "
        "Do not begin with phrases like 'This image is' or 'The picture shows'. "
        "Do not mention what is absent or outside the frame."
    )


def _tag_list(_: Optional[str], __: Optional[str], ___: Optional[int]) -> str:
    return "Provide a comma-separated list of concise tags that describe the image."


def _creative_prompt(_: Optional[str], __: Optional[str], ___: Optional[int]) -> str:
    return (
        "Write a rich, detailed text-to-image prompt that recreates the scene. "
        "Include subject, environment, lighting, camera perspective, and style details."
    )


PROMPTS: Dict[str, PromptSpec] = {
    "descriptive_long": PromptSpec(
        key="descriptive_long",
        label="Descriptive: long detailed",
        builder=_long_detailed,
    ),
    "descriptive_word_count": PromptSpec(
        key="descriptive_word_count",
        label="Descriptive: word count limit",
        builder=_detailed_word_count,
    ),
    "descriptive_length_tone": PromptSpec(
        key="descriptive_length_tone",
        label="Descriptive: length + tone",
        builder=_length_tone,
    ),
    "straightforward": PromptSpec(
        key="straightforward",
        label="Straightforward caption",
        builder=_straightforward,
    ),
    "tags": PromptSpec(
        key="tags",
        label="Tags list",
        builder=_tag_list,
    ),
    "creative_prompt": PromptSpec(
        key="creative_prompt",
        label="Creative text-to-image prompt",
        builder=_creative_prompt,
    ),
}


def build_prompt(
    mode: str,
    tone: Optional[str] = None,
    length: Optional[str] = None,
    word_count: Optional[int] = None,
) -> str:
    if mode not in PROMPTS:
        raise ValueError("Unsupported mode")
    return PROMPTS[mode].builder(length, tone, word_count)
