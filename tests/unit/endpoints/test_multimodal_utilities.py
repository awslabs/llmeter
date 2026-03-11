# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from llmeter.prompt_utils import (
    read_file,
    detect_format_from_extension,
    detect_format_from_bytes,
    detect_format_from_file,
)


class TestReadFile:
    """Test the read_file utility function."""

    def test_read_file_valid_path(self):
        """Test reading a valid file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content")
            temp_path = f.name

        try:
            content = read_file(temp_path)
            assert content == b"test content"
        finally:
            Path(temp_path).unlink()

    def test_read_file_nonexistent(self):
        """Test reading a non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            read_file("/nonexistent/file.txt")

    def test_read_file_binary_content(self):
        """Test reading binary content."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"\xff\xd8\xff\xe0")  # JPEG magic bytes
            temp_path = f.name

        try:
            content = read_file(temp_path)
            assert content == b"\xff\xd8\xff\xe0"
        finally:
            Path(temp_path).unlink()


class TestDetectFormatFromExtension:
    """Test the detect_format_from_extension utility function."""

    def test_detect_jpeg_extension(self):
        """Test detecting JPEG format from .jpg extension."""
        mime_type = detect_format_from_extension("image.jpg")
        assert mime_type == "image/jpeg"

    def test_detect_jpeg_extension_uppercase(self):
        """Test detecting JPEG format from .JPG extension."""
        mime_type = detect_format_from_extension("image.JPG")
        assert mime_type == "image/jpeg"

    def test_detect_png_extension(self):
        """Test detecting PNG format from .png extension."""
        mime_type = detect_format_from_extension("image.png")
        assert mime_type == "image/png"

    def test_detect_pdf_extension(self):
        """Test detecting PDF format from .pdf extension."""
        mime_type = detect_format_from_extension("document.pdf")
        assert mime_type == "application/pdf"

    def test_detect_mp4_extension(self):
        """Test detecting MP4 format from .mp4 extension."""
        mime_type = detect_format_from_extension("video.mp4")
        assert mime_type == "video/mp4"

    def test_detect_mp3_extension(self):
        """Test detecting MP3 format from .mp3 extension."""
        mime_type = detect_format_from_extension("audio.mp3")
        assert mime_type == "audio/mpeg"

    def test_detect_wav_extension(self):
        """Test detecting WAV format from .wav extension."""
        mime_type = detect_format_from_extension("audio.wav")
        assert mime_type == "audio/wav"

    def test_detect_unknown_extension(self):
        """Test detecting unknown extension returns None."""
        mime_type = detect_format_from_extension("file.unknown")
        assert mime_type is None

    def test_detect_no_extension(self):
        """Test detecting file without extension returns None."""
        mime_type = detect_format_from_extension("file")
        assert mime_type is None


class TestDetectFormatFromBytes:
    """Test the detect_format_from_bytes utility function."""

    def test_detect_jpeg_from_bytes_with_puremagic(self):
        """Test detecting JPEG format from bytes with puremagic."""
        jpeg_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"
        try:
            mime_type = detect_format_from_bytes(jpeg_bytes)
            # If puremagic is installed, it should detect the format
            if mime_type is not None:
                assert mime_type == "image/jpeg"
            else:
                # If puremagic is not installed, it returns None
                pytest.skip("puremagic not installed")
        except ImportError:
            pytest.skip("puremagic not installed")

    def test_detect_png_from_bytes_with_puremagic(self):
        """Test detecting PNG format from bytes with puremagic."""
        png_bytes = b"\x89PNG\r\n\x1a\n"
        try:
            mime_type = detect_format_from_bytes(png_bytes)
            # If puremagic is installed, it should detect the format
            if mime_type is not None:
                assert mime_type == "image/png"
            else:
                # If puremagic is not installed, it returns None
                pytest.skip("puremagic not installed")
        except ImportError:
            pytest.skip("puremagic not installed")

    def test_detect_format_without_puremagic(self):
        """Test that detection returns None when puremagic is not available."""
        # Mock puremagic to raise ImportError when accessed
        with patch("llmeter.prompt_utils.puremagic") as mock_puremagic:
            mock_puremagic.from_string.side_effect = ImportError("puremagic not available")
            mime_type = detect_format_from_bytes(b"\xff\xd8\xff\xe0")
            assert mime_type is None


class TestDetectFormatFromFile:
    """Test the detect_format_from_file utility function."""

    def test_detect_format_from_jpeg_file(self):
        """Test detecting format from JPEG file."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(
                b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"
            )
            temp_path = f.name

        try:
            mime_type = detect_format_from_file(temp_path)
            assert mime_type == "image/jpeg"
        finally:
            Path(temp_path).unlink()

    def test_detect_format_from_png_file(self):
        """Test detecting format from PNG file."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x89PNG\r\n\x1a\n")
            temp_path = f.name

        try:
            mime_type = detect_format_from_file(temp_path)
            assert mime_type == "image/png"
        finally:
            Path(temp_path).unlink()

    def test_detect_format_from_pdf_file(self):
        """Test detecting format from PDF file."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4\n")
            temp_path = f.name

        try:
            mime_type = detect_format_from_file(temp_path)
            assert mime_type == "application/pdf"
        finally:
            Path(temp_path).unlink()

    def test_detect_format_fallback_to_extension(self):
        """Test that detection falls back to extension when puremagic not installed."""
        # Create a file with JPEG magic bytes
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"\xff\xd8\xff\xe0")
            temp_path = f.name

        try:
            # Mock puremagic to raise ImportError when accessed
            with patch("llmeter.prompt_utils.puremagic") as mock_puremagic:
                mock_puremagic.magic_file.side_effect = ImportError("puremagic not available")
                mime_type = detect_format_from_file(temp_path)
                # Should fall back to extension-based detection
                assert mime_type == "image/jpeg"
        finally:
            Path(temp_path).unlink()

    def test_detect_format_no_extension_no_puremagic(self):
        """Test that detection returns None for file without extension when puremagic unavailable."""
        with tempfile.NamedTemporaryFile(suffix="", delete=False) as f:
            f.write(b"some content")
            temp_path = f.name

        try:
            # Mock puremagic to raise ImportError when accessed
            with patch("llmeter.prompt_utils.puremagic") as mock_puremagic:
                mock_puremagic.magic_file.side_effect = ImportError("puremagic not available")
                mime_type = detect_format_from_file(temp_path)
                # Should fall back to extension-based detection, which returns None for no extension
                assert mime_type is None
        finally:
            Path(temp_path).unlink()
