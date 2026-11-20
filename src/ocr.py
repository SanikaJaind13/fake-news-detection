"""OCR text extraction from images using Tesseract.

Supports English, Hindi, and Marathi (when language packs are installed).
Gracefully handles missing language packs, blurry images, and OCR errors.
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Optional

from PIL import Image
import pytesseract

from .errors import InputError, FakeNewsError


class OCRError(FakeNewsError):
    """OCR-specific errors (unsupported format, extraction failure, etc.)."""
    pass


@dataclass
class OCRResult:
    """Result of OCR extraction."""
    text: str
    language: str
    confidence: float  # average confidence across all words (0-100)
    raw_data: dict  # raw pytesseract output for debugging


# Supported language configurations
# Tesseract language codes: eng=English, hin=Hindi, mar=Marathi
SUPPORTED_LANGUAGES = {
    "English": "eng",
    "Hindi": "hin",
    "Marathi": "mar",
}

# Default language order: English first, then Hindi+Marathi
DEFAULT_LANG_COMBO = "eng+hin+mar"


def get_available_languages() -> list[str]:
    """Return list of available Tesseract language codes on the system."""
    try:
        langs = pytesseract.get_languages(config="")
        return [lang for lang in langs if lang != "osd"]
    except Exception:
        return ["eng"]  # fallback


def get_supported_language_codes() -> list[str]:
    """Return the language codes we support that are actually installed."""
    available = set(get_available_languages())
    return [code for name, code in SUPPORTED_LANGUAGES.items() if code in available]


def build_language_string(selected_languages: list[str] | None = None) -> str:
    """Build Tesseract language string from selected language names.

    Args:
        selected_languages: List of language names like ["English", "Hindi"]
                           If None, uses all available supported languages.

    Returns:
        Tesseract language string like "eng+hin+mar"
    """
    if selected_languages is None:
        # Use all available supported languages
        codes = get_supported_language_codes()
    else:
        codes = [SUPPORTED_LANGUAGES[lang] for lang in selected_languages if lang in SUPPORTED_LANGUAGES]

    # Filter to only installed languages
    available = set(get_available_languages())
    codes = [c for c in codes if c in available]

    if not codes:
        return "eng"  # fallback to English only

    return "+".join(codes)


def extract_text_from_image(
    image_bytes: bytes,
    languages: list[str] | None = None,
    config: str = "--psm 6 --oem 3",
) -> OCRResult:
    """Extract text from an image using Tesseract OCR.

    Args:
        image_bytes: Raw image bytes (PNG, JPG, JPEG)
        languages: Optional list of language names (e.g., ["English", "Hindi"])
        config: Tesseract config string (default: assume single uniform block of text)

    Returns:
        OCRResult with extracted text, language used, and confidence

    Raises:
        OCRError: If image format is unsupported or OCR fails
    """
    # Validate image format
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()  # Check if it's a valid image
        img = Image.open(io.BytesIO(image_bytes))  # Reopen after verify
    except Exception as exc:
        raise OCRError(f"Invalid or unsupported image format: {exc}")

    # Check format
    if img.format not in ("PNG", "JPEG", "JPG"):
        raise OCRError(f"Unsupported image format: {img.format}. Use PNG, JPG, or JPEG.")

    # Build language string
    lang_string = build_language_string(languages)

    try:
        # Get detailed OCR data including confidence
        data = pytesseract.image_to_data(
            img,
            lang=lang_string,
            config=config,
            output_type=pytesseract.Output.DICT,
        )

        # Extract text and calculate average confidence
        texts = []
        confidences = []

        for i in range(len(data["text"])):
            text = data["text"][i].strip()
            conf = data["conf"][i]
            if text and conf != -1:  # -1 means no confidence data
                texts.append(text)
                confidences.append(conf)

        if not texts:
            raise OCRError("No text could be extracted from the image. The image may be blurry, empty, or contain no readable text.")

        full_text = " ".join(texts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return OCRResult(
            text=full_text,
            language=lang_string,
            confidence=round(avg_confidence, 1),
            raw_data=data,
        )

    except pytesseract.TesseractError as exc:
        raise OCRError(f"Tesseract OCR failed: {exc}")
    except Exception as exc:
        raise OCRError(f"OCR extraction failed: {exc}")


def preprocess_image_for_ocr(image_bytes: bytes) -> bytes:
    """Optional: Preprocess image to improve OCR accuracy.

    Currently returns the original bytes. Can be extended with:
    - Grayscale conversion
    - Thresholding/binarization
    - Deskewing
    - Noise reduction
    - Resizing

    Args:
        image_bytes: Raw image bytes

    Returns:
        Preprocessed image bytes
    """
    # For now, return as-is. Can add PIL-based preprocessing later if needed.
    return image_bytes


def get_ocr_language_options() -> list[dict]:
    """Get language options for UI dropdown, marking unavailable ones."""
    available = set(get_available_languages())
    options = []
    for name, code in SUPPORTED_LANGUAGES.items():
        options.append({
            "name": name,
            "code": code,
            "available": code in available,
        })
    return options