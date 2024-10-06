# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from dataclasses import dataclass
from itertools import product
import os
import random
from typing import Any, Callable, Iterator

from upath import UPath as Path

from .tokenizers import DummyTokenizer, Tokenizer

logger = logging.getLogger(__name__)


@dataclass
class CreatePromptCollection:
    input_lengths: list[int]
    output_lengths: list[int]
    source_file: os.PathLike
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

    def _generate_sample(self, source_file: os.PathLike, sample_size: int) -> str:
        source_file = Path(source_file)
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
    file_path: os.PathLike,
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

    file_path = Path(file_path)
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


def load_payloads(file_path: os.PathLike | str) -> Iterator[dict]:
    """
    Load JSON payload(s) from a file or directory.

    This function reads JSON data from either a single file or multiple files
    in a directory. It supports both .json and .jsonl file formats.

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

    """
    file_path = Path(file_path)

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
                        yield json.loads(line.strip())
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON in {file}: {e}")
            else:  # Assume it's a regular JSON file
                yield json.load(f)
    except IOError as e:
        print(f"Error reading file {file}: {e}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in {file}: {e}")


def save_payloads(
    payloads: list[dict] | dict,
    output_path: os.PathLike | str,
    output_file: str = "payload.jsonl",
) -> Path:
    """
    Save payloads to a file.

    Args:
        payloads (Iterator[Dict]): An iterator of payloads (dicts).
        output_path (Union[Path, str]): The directory path where the output file should be saved.
        output_file (str, optional): The name of the output file. Defaults to "payloads.jsonl".

    Returns:
        output_file_path (UPath): The path to the output file.

    Raises:
        IOError: If there's an error writing to the file.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file_path = output_path / output_file

    if isinstance(payloads, dict):
        payloads = [payloads]
    with output_file_path.open(mode="w") as f:
        for payload in payloads:
            f.write(json.dumps(payload) + "\n")
    return output_file_path
