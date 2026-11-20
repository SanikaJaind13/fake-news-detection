"""Unit tests for the OCR module."""
import io
import unittest
from unittest.mock import patch, MagicMock

from PIL import Image

from src.errors import InputError
from src.ocr import (
    OCRResult,
    OCRError,
    extract_text_from_image,
    get_available_languages,
    get_supported_language_codes,
    build_language_string,
    get_ocr_language_options,
)


class TestOCRUtilities(unittest.TestCase):
    """Tests for OCR utility functions."""

    @patch("src.ocr.pytesseract.get_languages")
    def test_get_available_languages(self, mock_get_langs):
        mock_get_langs.return_value = ["eng", "hin", "mar", "osd"]
        langs = get_available_languages()
        self.assertIn("eng", langs)
        self.assertIn("hin", langs)
        self.assertIn("mar", langs)
        self.assertNotIn("osd", langs)  # OSD should be filtered out

    @patch("src.ocr.get_available_languages")
    def test_get_supported_language_codes(self, mock_get_avail):
        mock_get_avail.return_value = ["eng", "hin"]  # mar not installed
        codes = get_supported_language_codes()
        self.assertIn("eng", codes)
        self.assertIn("hin", codes)
        self.assertNotIn("mar", codes)

    @patch("src.ocr.get_supported_language_codes")
    def test_build_language_string_default(self, mock_get_codes):
        mock_get_codes.return_value = ["eng", "hin", "mar"]
        lang_str = build_language_string()
        self.assertEqual(lang_str, "eng+hin+mar")

    @patch("src.ocr.get_supported_language_codes")
    def test_build_language_string_selected(self, mock_get_codes):
        mock_get_codes.return_value = ["eng", "hin", "mar"]
        lang_str = build_language_string(["English", "Marathi"])
        self.assertEqual(lang_str, "eng+mar")

    @patch("src.ocr.get_supported_language_codes")
    def test_build_language_string_empty_fallback(self, mock_get_codes):
        mock_get_codes.return_value = []
        lang_str = build_language_string()
        self.assertEqual(lang_str, "eng")  # fallback

    @patch("src.ocr.get_available_languages")
    def test_get_ocr_language_options(self, mock_get_avail):
        mock_get_avail.return_value = ["eng", "hin"]
        options = get_ocr_language_options()
        self.assertEqual(len(options), 3)
        # English available
        self.assertTrue(options[0]["available"])
        self.assertEqual(options[0]["name"], "English")
        # Hindi available
        self.assertTrue(options[1]["available"])
        self.assertEqual(options[1]["name"], "Hindi")
        # Marathi not available
        self.assertFalse(options[2]["available"])
        self.assertEqual(options[2]["name"], "Marathi")


class TestExtractTextFromImage(unittest.TestCase):
    """Tests for the main OCR extraction function."""

    def _create_test_image(self, text="Test", color=(255, 255, 255)):
        """Create a simple test image with text."""
        img = Image.new("RGB", (400, 100), color=color)
        return img

    def _image_to_bytes(self, img, format="PNG"):
        """Convert PIL image to bytes."""
        buf = io.BytesIO()
        img.save(buf, format=format)
        return buf.getvalue()

    @patch("src.ocr.pytesseract.image_to_data")
    def test_extract_text_success(self, mock_image_to_data):
        """Test successful OCR extraction with mocked tesseract."""
        # Mock the pytesseract output
        mock_image_to_data.return_value = {
            "text": ["Hello", "world", "", "test"],
            "conf": [95, 90, -1, 85],
        }

        img = self._create_test_image()
        img_bytes = self._image_to_bytes(img)

        result = extract_text_from_image(img_bytes)

        self.assertIsInstance(result, OCRResult)
        self.assertEqual(result.text, "Hello world test")
        self.assertAlmostEqual(result.confidence, 90.0, places=1)
        self.assertIn("eng", result.language)

    @patch("src.ocr.pytesseract.image_to_data")
    def test_extract_text_empty_result(self, mock_image_to_data):
        """Test OCR when no text is extracted."""
        mock_image_to_data.return_value = {
            "text": ["", "", ""],
            "conf": [-1, -1, -1],
        }

        img = self._create_test_image()
        img_bytes = self._image_to_bytes(img)

        with self.assertRaises(OCRError) as ctx:
            extract_text_from_image(img_bytes)
        self.assertIn("No text could be extracted", str(ctx.exception))

    @patch("src.ocr.pytesseract.image_to_data")
    def test_extract_text_tesseract_error(self, mock_image_to_data):
        """Test OCR when tesseract raises an error."""
        import pytesseract
        mock_image_to_data.side_effect = pytesseract.TesseractError(1, "Tesseract failed")

        img = self._create_test_image()
        img_bytes = self._image_to_bytes(img)

        with self.assertRaises(OCRError) as ctx:
            extract_text_from_image(img_bytes)
        self.assertIn("Tesseract OCR failed", str(ctx.exception))

    def test_extract_text_invalid_image(self):
        """Test OCR with invalid image bytes."""
        with self.assertRaises(OCRError) as ctx:
            extract_text_from_image(b"not an image")
        self.assertIn("Invalid or unsupported image format", str(ctx.exception))

    def test_extract_text_unsupported_format(self):
        """Test OCR with unsupported image format (e.g., BMP)."""
        img = self._create_test_image()
        # Save as BMP
        buf = io.BytesIO()
        img.save(buf, format="BMP")
        bmp_bytes = buf.getvalue()

        with self.assertRaises(OCRError) as ctx:
            extract_text_from_image(bmp_bytes)
        self.assertIn("Unsupported image format: BMP", str(ctx.exception))

    def test_extract_text_jpg_format(self):
        """Test OCR accepts JPEG format."""
        img = self._create_test_image()
        jpg_bytes = self._image_to_bytes(img, format="JPEG")

        with patch("src.ocr.pytesseract.image_to_data") as mock:
            mock.return_value = {"text": ["test"], "conf": [90]}
            result = extract_text_from_image(jpg_bytes)
            self.assertEqual(result.text, "test")

    def test_extract_text_png_format(self):
        """Test OCR accepts PNG format."""
        img = self._create_test_image()
        png_bytes = self._image_to_bytes(img, format="PNG")

        with patch("src.ocr.pytesseract.image_to_data") as mock:
            mock.return_value = {"text": ["test"], "conf": [90]}
            result = extract_text_from_image(png_bytes)
            self.assertEqual(result.text, "test")

    def test_extract_text_with_language_selection(self):
        """Test OCR with specific language selection."""
        img = self._create_test_image()
        img_bytes = self._image_to_bytes(img)

        with patch("src.ocr.pytesseract.image_to_data") as mock:
            mock.return_value = {"text": ["नमस्ते"], "conf": [85]}
            with patch("src.ocr.get_supported_language_codes") as mock_codes:
                mock_codes.return_value = ["eng", "hin"]
                result = extract_text_from_image(img_bytes, languages=["Hindi"])
                self.assertEqual(result.language, "hin")


class TestOCRIntegration(unittest.TestCase):
    """Integration-style tests (require tesseract to be installed)."""

    @unittest.skipUnless(
        __import__("shutil").which("tesseract"),
        "Tesseract not installed, skipping integration test"
    )
    def test_real_ocr_extraction(self):
        """Test real OCR extraction on a simple image."""
        # Create an image with clear text
        img = Image.new("RGB", (400, 100), color=(255, 255, 255))
        # We can't easily draw text without additional deps, so this test
        # mainly verifies the function runs without crashing
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        # This should either work or fail gracefully
        try:
            result = extract_text_from_image(img_bytes)
            self.assertIsInstance(result, OCRResult)
            self.assertIn(result.language, ["eng", "eng+hin", "eng+hin+mar", "eng+mar", "hin", "mar"])
        except OCRError:
            # Empty image might not have text - that's ok
            pass


if __name__ == "__main__":
    unittest.main()