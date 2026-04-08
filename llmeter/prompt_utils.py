# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import random
from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Iterator

from upath import UPath as Path
from upath.types import ReadablePathLike, WritablePathLike

from .json_utils import LLMeterEncoder, llmeter_bytes_decoder
from .tokenizers import DummyTokenizer, Tokenizer
from .utils import DeferredError, ensure_path

logger = logging.getLogger(__name__)

# Optional dependency: puremagic for content-based format detection
try:
    import puremagic
except ImportError as e:
    logger.debug(
        "puremagic not available. Format detection will fall back to file extensions. "
        "Install with: pip install 'llmeter[multimodal]'"
    )
    puremagic = DeferredError(e)


# Multi-modal content utilities


def read_file(file_path: ReadablePathLike) -> bytes:
    """Read binary content from a file.

    Args:
        file_path: Path to the file

    Returns:
        bytes: File content

    Raises:
        FileNotFoundError: If file doesn't exist
        IOError: If file cannot be read
    """
    _path = ensure_path(file_path)
    with _path.open("rb") as f:
        return f.read()


def detect_format_from_extension(file_path: ReadablePathLike) -> str | None:
    """Detect MIME type from file extension.

    Args:
        file_path: Path to the file

    Returns:
        str | None: MIME type or None if extension not recognized

    Examples:
        >>> detect_format_from_extension("image.jpg")
        "image/jpeg"
        >>> detect_format_from_extension("document.pdf")
        "application/pdf"
    """
    extension = ensure_path(file_path).suffix.lower()

    # Map common extensions to MIME types
    extension_to_mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".pdf": "application/pdf",
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".ogg": "audio/ogg",
    }

    return extension_to_mime.get(extension)


def detect_format_from_bytes(content: bytes) -> str | None:
    """Detect MIME type from bytes content using puremagic.

    Args:
        content: Binary content

    Returns:
        str | None: MIME type or None if detection fails or puremagic not available

    Examples:
        >>> detect_format_from_bytes(b"\\xff\\xd8\\xff\\xe0")  # JPEG magic bytes
        "image/jpeg"
    """
    try:
        # Get MIME type from content using puremagic (v2.0+ API)
        mime_type = puremagic.from_string(content, mime=True)
        return mime_type if mime_type else None
    except (ImportError, AttributeError):
        # puremagic not available or DeferredError raised
        return None
    except Exception:
        pass

    return None


def detect_format_from_file(file_path: ReadablePathLike) -> str | None:
    """Detect MIME type from file using puremagic or extension fallback.

    Args:
        file_path: Path to the file

    Returns:
        str | None: MIME type or None if format cannot be detected

    Examples:
        >>> detect_format_from_file("photo.jpg")
        "image/jpeg"
    """
    # Try puremagic first if available
    try:
        matches = puremagic.magic_file(file_path)
        if matches:
            # Extract MIME type from first match
            mime_type = (
                matches[0].mime_type if hasattr(matches[0], "mime_type") else None
            )
            if mime_type:
                return mime_type
    except (ImportError, AttributeError):
        # puremagic not available or DeferredError raised
        pass
    except Exception:
        pass

    # Fallback to extension-based detection
    return detect_format_from_extension(file_path)


@dataclass
class CreatePromptCollection:
    input_lengths: list[int]
    output_lengths: list[int]
    source_file: ReadablePathLike
    requests_per_combination: int = 1
    tokenizer: Tokenizer | None = None
    source_file_encoding: str = "utf-8-sig"

    def __post_init__(self):
        self._tokenizer = self.tokenizer or DummyTokenizer()

    def create_collection(self) -> list[Any]:
        self._generate_samples()
        collection = (
            list(product(self._samples, self.output_lengths))
            * self.requests_per_combination
        )
        return random.sample(collection, k=len(collection))

    def _generate_sample(self, source_file: ReadablePathLike, sample_size: int) -> str:
        source_file = ensure_path(source_file)
        sample = []
        with source_file.open(encoding=self.source_file_encoding, mode="r") as f:
            for line in f:
                encoded_line = self._tokenizer.encode(line)
                sample += encoded_line
                if len(sample) > sample_size:
                    break
        return self._tokenizer.decode(sample[:sample_size])

    def _generate_samples(self) -> None:
        self._samples = [
            self._generate_sample(self.source_file, k) for k in self.input_lengths
        ]


def load_prompts(
    file_path: ReadablePathLike,
    create_payload_fn: Callable,
    create_payload_kwargs: dict = {},
    file_pattern: str | None = None,
) -> Iterator[dict]:
    """
    Load prompts from a file or directory and create payloads.

    This function reads prompts from either a single file or multiple files in a directory,
    and generates payloads using the provided create_payload_fn.

    Args:
        file_path (Union[UPath, str]): Path to a file or directory containing prompts.
        create_payload_fn (Callable): Function to create a payload from each prompt.
        create_payload_kwargs (Dict, optional): Additional keyword arguments for create_payload_fn.
            Defaults to an empty dictionary.
        file_pattern (Union[str, None], optional): Glob pattern for matching files in a directory.
            If None, matches all files. Defaults to None.

    Yields:
        Dict: Payload created from each prompt.

    Raises:
        FileNotFoundError: If the specified file or directory does not exist.
        PermissionError: If there's insufficient permission to read the file(s).
        ValueError: If create_payload_fn raises a ValueError.

    """

    file_path = ensure_path(file_path)
    if file_path.is_file():
        with file_path.open(mode="r") as f:
            for line in f:
                if not line.strip():
                    continue
                yield create_payload_fn(
                    input_text=line.strip(), **create_payload_kwargs
                )
    for file in file_path.glob(file_pattern or "*"):
        with file.open(mode="r") as f:
            for line in f:
                try:
                    if not line.strip():
                        continue

                    yield create_payload_fn(
                        input_text=line.strip(), **create_payload_kwargs
                    )
                except Exception as e:
                    logger.exception(f"Error processing line: {line}: {e}")
                    continue


def load_payloads(
    file_path: ReadablePathLike,
) -> Iterator[dict]:
    """
    Load JSON payload(s) from a file or directory with binary content support.

    This function reads JSON data from either a single file or multiple files
    in a directory. It supports both .json and .jsonl file formats. Binary content
    (bytes objects) that were serialized using LLMeterEncoder are automatically
    restored during deserialization.

    Binary Content Handling:
        When loading payloads saved with save_payloads(), marker objects with the key
        "__llmeter_bytes__" are automatically detected and converted back to bytes objects.
        The base64-encoded strings are decoded to restore the original binary data,
        enabling round-trip preservation of multimodal content like images and video.

        The marker object format is: {"__llmeter_bytes__": "<base64-string>"}

    Args:
        file_path (Union[Path, str]): Path to a JSON file or a directory
            containing JSON files. Can be a string or a Path object.

    Yields:
        dict: Each JSON object loaded from the file(s).

    Raises:
        FileNotFoundError: If the specified file or directory does not exist.
        json.JSONDecodeError: If there's an error parsing the JSON data.
        ValidationError: If the JSON data does not conform to the expected schema.
        IOError: If there's an error reading the file.

    Examples:
        Load a Bedrock Converse API payload with image content:

        >>> # Assuming a file was saved with save_payloads() containing binary data
        >>> payloads = list(load_payloads("/tmp/output/payload.jsonl"))
        >>> payload = payloads[0]
        >>> # Binary content is automatically restored as bytes
        >>> image_bytes = payload["messages"][0]["content"][1]["image"]["source"]["bytes"]
        >>> isinstance(image_bytes, bytes)
        True
        >>> # The bytes can be used directly with the API
        >>> print(f"Image size: {len(image_bytes)} bytes")
        Image size: 52341 bytes

        Load multiple payloads with video content:

        >>> for payload in load_payloads("/tmp/output/multimodal.jsonl"):
        ...     video_content = payload["messages"][0]["content"][1]
        ...     if "video" in video_content:
        ...         video_bytes = video_content["video"]["source"]["bytes"]
        ...         print(f"Loaded video: {len(video_bytes)} bytes")
        Loaded video: 1048576 bytes

        Load all payloads from a directory:

        >>> # Load all .json and .jsonl files in a directory
        >>> all_payloads = list(load_payloads("/tmp/output/"))
        >>> print(f"Loaded {len(all_payloads)} payloads")
        Loaded 5 payloads

        Round-trip example showing binary preservation:

        >>> # Original payload with binary data
        >>> original = {
        ...     "modelId": "test-model",
        ...     "messages": [{
        ...         "role": "user",
        ...         "content": [
        ...             {"image": {"source": {"bytes": b"\\xff\\xd8\\xff\\xe0"}}}
        ...         ]
        ...     }]
        ... }
        >>> # Save and load
        >>> save_payloads(original, "/tmp/test")
        PosixPath('/tmp/test/payload.jsonl')
        >>> loaded = list(load_payloads("/tmp/test/payload.jsonl"))[0]
        >>> # Binary data is preserved byte-for-byte
        >>> original == loaded
        True
    """
    file_path = ensure_path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"The specified path does not exist: {file_path}")

    if file_path.is_file():
        yield from _load_data_file(file_path)
    else:
        for file in file_path.glob("*.json*"):
            yield from _load_data_file(file)


def _load_data_file(file: Path) -> Iterator[dict]:
    try:
        with file.open(mode="r") as f:
            if file.suffix.lower() in [".jsonl", ".manifest"]:
                for line in f:
                    try:
                        if not line.strip():
                            continue
                        yield json.loads(
                            line.strip(), object_hook=llmeter_bytes_decoder
                        )
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON in {file}: {e}")
            else:  # Assume it's a regular JSON file
                yield json.load(f, object_hook=llmeter_bytes_decoder)
    except IOError as e:
        print(f"Error reading file {file}: {e}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in {file}: {e}")


def save_payloads(
    payloads: list[dict] | dict,
    output_path: WritablePathLike,
    output_file: str = "payload.jsonl",
) -> Path:
    """
    Save payloads to a file with support for binary content.

    This function saves payloads to a JSONL file, with automatic handling of binary
    content (bytes objects) through base64 encoding. Binary data is wrapped in marker
    objects during serialization to enable round-trip preservation.

    Binary Content Handling:
        When a payload contains bytes objects (e.g., images, video), they are automatically
        converted to base64-encoded strings and wrapped in a marker object with the key
        "__llmeter_bytes__". This approach enables JSON serialization while preserving
        the ability to restore the original bytes during deserialization with load_payloads().

        The marker object format is: {"__llmeter_bytes__": "<base64-string>"}

    Args:
        payloads (Union[list[dict], dict]): Payload(s) to save. May contain bytes objects
            at any nesting level.
        output_path (Union[Path, str]): The directory path where the output file should be saved.
        output_file (str, optional): The name of the output file. Defaults to "payload.jsonl".

    Returns:
        Path: The path to the output file.

    Raises:
        IOError: If there's an error writing to the file.
        TypeError: If payload contains unserializable types.

    Examples:
        Save a Bedrock Converse API payload with image content:

        >>> import base64
        >>> # Create a payload with binary image data
        >>> with open("image.jpg", "rb") as f:
        ...     image_bytes = f.read()
        >>> payload = {
        ...     "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
        ...     "messages": [{
        ...         "role": "user",
        ...         "content": [
        ...             {"text": "What is in this image?"},
        ...             {
        ...                 "image": {
        ...                     "format": "jpeg",
        ...                     "source": {"bytes": image_bytes}
        ...                 }
        ...             }
        ...         ]
        ...     }]
        ... }
        >>> output_path = save_payloads(payload, "/tmp/output")
        >>> print(output_path)
        /tmp/output/payload.jsonl

        Save multiple payloads with video content:

        >>> with open("video.mp4", "rb") as f:
        ...     video_bytes = f.read()
        >>> payloads = [
        ...     {
        ...         "modelId": "anthropic.claude-3-sonnet-20240229-v1:0",
        ...         "messages": [{
        ...             "role": "user",
        ...             "content": [
        ...                 {"text": "Describe this video"},
        ...                 {
        ...                     "video": {
        ...                         "format": "mp4",
        ...                         "source": {"bytes": video_bytes}
        ...                     }
        ...                 }
        ...             ]
        ...         }]
        ...     }
        ... ]
        >>> save_payloads(payloads, "/tmp/output", "multimodal.jsonl")
        PosixPath('/tmp/output/multimodal.jsonl')

        The saved JSON file will contain marker objects for binary data:

        >>> # Example of what gets written to the file:
        >>> # {
        >>> #   "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
        >>> #   "messages": [{
        >>> #     "role": "user",
        >>> #     "content": [
        >>> #       {"text": "What is in this image?"},
        >>> #       {
        >>> #         "image": {
        >>> #           "format": "jpeg",
        >>> #           "source": {
        >>> #             "bytes": {"__llmeter_bytes__": "/9j/4AAQSkZJRg..."}
        >>> #           }
        >>> #         }
        >>> #       }
        >>> #     ]
        >>> #   }]
        >>> # }
    """
    output_path = ensure_path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file_path = output_path / output_file

    if isinstance(payloads, dict):
        payloads = [payloads]
    with output_file_path.open(mode="w") as f:
        for payload in payloads:
            f.write(json.dumps(payload, cls=LLMeterEncoder) + "\n")
    return output_file_path
