# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for Boto3S3FileSystem against real S3.

These tests require:
- Valid AWS credentials with S3 read/write permissions
- S3FS_TEST_BUCKET environment variable set to a writable bucket

Run with: uv run pytest -m integ tests/integ/test_s3fs_integ.py

Requirements validated:
- R1-AC3: UPath integration without s3fs installed
- R2-AC1-4: File read operations
- R3-AC1-4: File write operations
- R4-AC2,3,5,6: Directory operations (_ls, _info)
- R5-AC1-3: Existence checks
- R11-AC1,2: Round-trip data integrity
- R12-AC1,2: Async execution model (to_thread, concurrent gather)
"""

from __future__ import annotations

import asyncio

import pytest

from llmeter.s3fs import Boto3S3FileSystem

pytestmark = pytest.mark.integ


# --- Helper to build S3 paths ---


def s3_path(bucket: str, prefix: str, key: str) -> str:
    """Build a full S3 path (without protocol) from bucket, prefix, and key."""
    return f"{bucket}/{prefix}{key}"


# --- Round-trip write/read tests ---


class TestRoundTrip:
    """Test round-trip write/read with real S3 (binary and text, various sizes)."""

    @pytest.mark.asyncio
    async def test_roundtrip_binary_small(
        self, s3_test_bucket, s3_test_prefix, aws_credentials
    ):
        """Write and read back small binary content (100 bytes)."""
        fs = Boto3S3FileSystem(session=aws_credentials)
        path = s3_path(s3_test_bucket, s3_test_prefix, "roundtrip/small.bin")
        data = b"x" * 100

        await fs._pipe_file(path, data)
        result = await fs._cat_file(path)

        assert result == data

    @pytest.mark.asyncio
    async def test_roundtrip_binary_empty(
        self, s3_test_bucket, s3_test_prefix, aws_credentials
    ):
        """Write and read back empty content (0 bytes)."""
        fs = Boto3S3FileSystem(session=aws_credentials)
        path = s3_path(s3_test_bucket, s3_test_prefix, "roundtrip/empty.bin")
        data = b""

        await fs._pipe_file(path, data)
        result = await fs._cat_file(path)

        assert result == data

    @pytest.mark.asyncio
    async def test_roundtrip_binary_medium(
        self, s3_test_bucket, s3_test_prefix, aws_credentials
    ):
        """Write and read back medium binary content (1MB)."""
        fs = Boto3S3FileSystem(session=aws_credentials)
        path = s3_path(s3_test_bucket, s3_test_prefix, "roundtrip/medium.bin")
        data = b"\xab\xcd" * (1024 * 512)  # 1MB

        await fs._pipe_file(path, data)
        result = await fs._cat_file(path)

        assert result == data
        assert len(result) == 1024 * 1024

    @pytest.mark.asyncio
    async def test_roundtrip_text(
        self, s3_test_bucket, s3_test_prefix, aws_credentials
    ):
        """Write and read back text content via open() modes."""
        fs = Boto3S3FileSystem(session=aws_credentials)
        path = s3_path(s3_test_bucket, s3_test_prefix, "roundtrip/text.txt")
        text = "Hello, world!\nLine 2\nUnicode: café résumé naïve"

        # Write text
        with fs.open(path, "w") as f:
            f.write(text)

        # Read text
        with fs.open(path, "r") as f:
            result = f.read()

        assert result == text

    @pytest.mark.asyncio
    async def test_roundtrip_binary_via_open(
        self, s3_test_bucket, s3_test_prefix, aws_credentials
    ):
        """Write and read back binary content via open() modes."""
        fs = Boto3S3FileSystem(session=aws_credentials)
        path = s3_path(s3_test_bucket, s3_test_prefix, "roundtrip/binary_open.bin")
        data = bytes(range(256)) * 4

        with fs.open(path, "wb") as f:
            f.write(data)

        with fs.open(path, "rb") as f:
            result = f.read()

        assert result == data


# --- Directory listing tests ---


class TestListing:
    """Test _ls listing with real S3 objects."""

    @pytest.mark.asyncio
    async def test_ls_detail_false(
        self, s3_test_bucket, s3_test_prefix, aws_credentials
    ):
        """_ls with detail=False returns list of path strings."""
        fs = Boto3S3FileSystem(session=aws_credentials)
        prefix = f"{s3_test_prefix}ls_test/"

        # Create test objects
        await fs._pipe_file(f"{s3_test_bucket}/{prefix}file1.txt", b"a")
        await fs._pipe_file(f"{s3_test_bucket}/{prefix}file2.txt", b"b")
        await fs._pipe_file(f"{s3_test_bucket}/{prefix}subdir/file3.txt", b"c")

        # List
        result = await fs._ls(f"{s3_test_bucket}/{prefix.rstrip('/')}", detail=False)

        # Should contain file1, file2, and subdir/ as immediate children
        names = [r.split("/")[-1] for r in result]
        assert "file1.txt" in names
        assert "file2.txt" in names
        assert "subdir" in names

    @pytest.mark.asyncio
    async def test_ls_detail_true(
        self, s3_test_bucket, s3_test_prefix, aws_credentials
    ):
        """_ls with detail=True returns list of metadata dicts."""
        fs = Boto3S3FileSystem(session=aws_credentials)
        prefix = f"{s3_test_prefix}ls_detail/"

        await fs._pipe_file(f"{s3_test_bucket}/{prefix}data.json", b'{"key": "value"}')

        result = await fs._ls(f"{s3_test_bucket}/{prefix.rstrip('/')}", detail=True)

        assert len(result) >= 1
        entry = next(e for e in result if e["name"].endswith("data.json"))
        assert entry["type"] == "file"
        assert entry["size"] == len(b'{"key": "value"}')
        assert "name" in entry

    @pytest.mark.asyncio
    async def test_ls_bare_bucket(
        self, s3_test_bucket, s3_test_prefix, aws_credentials
    ):
        """_ls on a prefix that has objects returns results."""
        fs = Boto3S3FileSystem(session=aws_credentials)

        # Ensure at least one object exists
        await fs._pipe_file(
            f"{s3_test_bucket}/{s3_test_prefix}ls_bare/marker.txt", b"exists"
        )

        # List the test prefix (simulates listing under a "directory")
        result = await fs._ls(
            f"{s3_test_bucket}/{s3_test_prefix.rstrip('/')}", detail=False
        )
        assert len(result) > 0


# --- Info and exists tests ---


class TestInfoExists:
    """Test _info and _exists with real S3."""

    @pytest.mark.asyncio
    async def test_info_file(self, s3_test_bucket, s3_test_prefix, aws_credentials):
        """_info on an existing object returns file metadata."""
        fs = Boto3S3FileSystem(session=aws_credentials)
        path = s3_path(s3_test_bucket, s3_test_prefix, "info/file.txt")
        content = b"hello info"

        await fs._pipe_file(path, content)
        info = await fs._info(path)

        assert info["type"] == "file"
        assert info["size"] == len(content)
        assert info["name"] == path

    @pytest.mark.asyncio
    async def test_info_directory(
        self, s3_test_bucket, s3_test_prefix, aws_credentials
    ):
        """_info on a prefix with children returns directory metadata."""
        fs = Boto3S3FileSystem(session=aws_credentials)
        prefix = f"{s3_test_prefix}info_dir"

        # Create a child object
        await fs._pipe_file(f"{s3_test_bucket}/{prefix}/child.txt", b"child")

        info = await fs._info(f"{s3_test_bucket}/{prefix}")
        assert info["type"] == "directory"

    @pytest.mark.asyncio
    async def test_info_not_found(
        self, s3_test_bucket, s3_test_prefix, aws_credentials
    ):
        """_info on a non-existent path raises FileNotFoundError."""
        fs = Boto3S3FileSystem(session=aws_credentials)
        path = s3_path(s3_test_bucket, s3_test_prefix, "info/nonexistent.txt")

        with pytest.raises(FileNotFoundError):
            await fs._info(path)

    @pytest.mark.asyncio
    async def test_exists_true(self, s3_test_bucket, s3_test_prefix, aws_credentials):
        """_exists returns True for existing objects."""
        fs = Boto3S3FileSystem(session=aws_credentials)
        path = s3_path(s3_test_bucket, s3_test_prefix, "exists/file.txt")

        await fs._pipe_file(path, b"exists")
        assert await fs._exists(path) is True

    @pytest.mark.asyncio
    async def test_exists_false(self, s3_test_bucket, s3_test_prefix, aws_credentials):
        """_exists returns False for non-existent paths."""
        fs = Boto3S3FileSystem(session=aws_credentials)
        path = s3_path(s3_test_bucket, s3_test_prefix, "exists/nope.txt")

        assert await fs._exists(path) is False

    @pytest.mark.asyncio
    async def test_exists_prefix(self, s3_test_bucket, s3_test_prefix, aws_credentials):
        """_exists returns True for non-empty prefixes."""
        fs = Boto3S3FileSystem(session=aws_credentials)
        prefix = f"{s3_test_prefix}exists_prefix"

        await fs._pipe_file(f"{s3_test_bucket}/{prefix}/child.txt", b"data")
        assert await fs._exists(f"{s3_test_bucket}/{prefix}") is True


# --- Removal tests ---


class TestRemoval:
    """Test _rm and recursive deletion on real S3."""

    @pytest.mark.asyncio
    async def test_rm_single_file(
        self, s3_test_bucket, s3_test_prefix, aws_credentials
    ):
        """_rm_file deletes a single object."""
        fs = Boto3S3FileSystem(session=aws_credentials)
        path = s3_path(s3_test_bucket, s3_test_prefix, "rm/single.txt")

        await fs._pipe_file(path, b"to delete")
        assert await fs._exists(path) is True

        await fs._rm_file(path)
        assert await fs._exists(path) is False

    @pytest.mark.asyncio
    async def test_rm_nonexistent_raises(
        self, s3_test_bucket, s3_test_prefix, aws_credentials
    ):
        """_rm_file on non-existent path raises FileNotFoundError."""
        fs = Boto3S3FileSystem(session=aws_credentials)
        path = s3_path(s3_test_bucket, s3_test_prefix, "rm/ghost.txt")

        with pytest.raises(FileNotFoundError):
            await fs._rm_file(path)

    @pytest.mark.asyncio
    async def test_rm_recursive(self, s3_test_bucket, s3_test_prefix, aws_credentials):
        """_rm with recursive=True deletes all objects under a prefix."""
        fs = Boto3S3FileSystem(session=aws_credentials)
        prefix = f"{s3_test_prefix}rm_recursive"
        base = f"{s3_test_bucket}/{prefix}"

        # Create multiple objects
        await fs._pipe_file(f"{base}/a.txt", b"a")
        await fs._pipe_file(f"{base}/b.txt", b"b")
        await fs._pipe_file(f"{base}/sub/c.txt", b"c")

        # Recursive delete
        fs.rm(f"{base}", recursive=True)

        # Verify all are gone
        assert await fs._exists(f"{base}/a.txt") is False
        assert await fs._exists(f"{base}/b.txt") is False
        assert await fs._exists(f"{base}/sub/c.txt") is False


# --- UPath integration tests ---


class TestUPathIntegration:
    """Test UPath integration with real S3 bucket."""

    def test_upath_write_text_read_text(
        self, s3_test_bucket, s3_test_prefix, aws_credentials
    ):
        """UPath read_text/write_text round-trip."""
        from upath import UPath

        path = UPath(
            f"s3://{s3_test_bucket}/{s3_test_prefix}upath/text.txt",
            session=aws_credentials,
        )
        text = "Hello from UPath!\nLine 2"

        path.write_text(text)
        result = path.read_text()

        assert result == text

    def test_upath_iterdir(self, s3_test_bucket, s3_test_prefix, aws_credentials):
        """UPath iterdir lists directory contents."""
        from upath import UPath

        base = UPath(
            f"s3://{s3_test_bucket}/{s3_test_prefix}upath_iterdir",
            session=aws_credentials,
        )

        # Create files
        (base / "one.txt").write_text("one")
        (base / "two.txt").write_text("two")

        # Iterdir
        names = sorted(p.name for p in base.iterdir())
        assert "one.txt" in names
        assert "two.txt" in names

    def test_upath_glob(self, s3_test_bucket, s3_test_prefix, aws_credentials):
        """UPath glob matches patterns."""
        from upath import UPath

        base = UPath(
            f"s3://{s3_test_bucket}/{s3_test_prefix}upath_glob",
            session=aws_credentials,
        )

        # Create files with different extensions
        (base / "data.json").write_text('{"a": 1}')
        (base / "data.csv").write_text("a,b,c")
        (base / "sub" / "nested.json").write_text('{"b": 2}')

        # Glob for json files (non-recursive)
        json_files = sorted(p.name for p in base.glob("*.json"))
        assert "data.json" in json_files
        assert "data.csv" not in json_files

        # Recursive glob
        all_json = sorted(p.name for p in base.glob("**/*.json"))
        assert "data.json" in all_json
        assert "nested.json" in all_json


# --- Concurrent async operations ---


class TestConcurrentAsync:
    """Test concurrent async operations via asyncio.gather on real S3."""

    @pytest.mark.asyncio
    async def test_concurrent_writes(
        self, s3_test_bucket, s3_test_prefix, aws_credentials
    ):
        """Multiple concurrent pipe_file operations complete successfully."""
        fs = Boto3S3FileSystem(session=aws_credentials)
        prefix = f"{s3_test_prefix}concurrent/writes"

        tasks = [
            fs._pipe_file(
                f"{s3_test_bucket}/{prefix}/file_{i}.txt",
                f"content_{i}".encode(),
            )
            for i in range(10)
        ]

        await asyncio.gather(*tasks)

        # Verify all files exist with correct content
        for i in range(10):
            path = f"{s3_test_bucket}/{prefix}/file_{i}.txt"
            data = await fs._cat_file(path)
            assert data == f"content_{i}".encode()

    @pytest.mark.asyncio
    async def test_concurrent_reads(
        self, s3_test_bucket, s3_test_prefix, aws_credentials
    ):
        """Multiple concurrent cat_file operations complete successfully."""
        fs = Boto3S3FileSystem(session=aws_credentials)
        prefix = f"{s3_test_prefix}concurrent/reads"

        # Write files first
        for i in range(10):
            await fs._pipe_file(
                f"{s3_test_bucket}/{prefix}/file_{i}.txt",
                f"read_content_{i}".encode(),
            )

        # Read all concurrently
        tasks = [
            fs._cat_file(f"{s3_test_bucket}/{prefix}/file_{i}.txt") for i in range(10)
        ]
        results = await asyncio.gather(*tasks)

        for i, result in enumerate(results):
            assert result == f"read_content_{i}".encode()

    @pytest.mark.asyncio
    async def test_concurrent_mixed_operations(
        self, s3_test_bucket, s3_test_prefix, aws_credentials
    ):
        """Mixed concurrent operations (write, read, exists) work correctly."""
        fs = Boto3S3FileSystem(session=aws_credentials)
        prefix = f"{s3_test_prefix}concurrent/mixed"

        # Write a file first
        path = f"{s3_test_bucket}/{prefix}/target.txt"
        await fs._pipe_file(path, b"target data")

        # Run mixed operations concurrently
        results = await asyncio.gather(
            fs._cat_file(path),
            fs._exists(path),
            fs._info(path),
            fs._pipe_file(f"{s3_test_bucket}/{prefix}/new.txt", b"new"),
        )

        assert results[0] == b"target data"
        assert results[1] is True
        assert results[2]["type"] == "file"
