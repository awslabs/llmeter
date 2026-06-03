# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Boto3S3FileSystem using moto mock_aws."""

from __future__ import annotations

import asyncio
import sys
import types
from unittest.mock import patch

import boto3
import pytest
from moto import mock_aws

from llmeter.s3fs import Boto3S3FileSystem

BUCKET = "test-bucket"


# --- Test _cat_file / _pipe_file round-trip ---


class TestCatFilePipeFile:
    """Tests for _cat_file and _pipe_file round-trip operations."""

    @mock_aws
    def test_roundtrip_binary(self):
        """Binary data round-trips correctly."""
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        fs = Boto3S3FileSystem()
        data = b"\x00\x01\x02\xff" * 100
        fs.pipe_file(f"{BUCKET}/binary.dat", data)
        result = fs.cat_file(f"{BUCKET}/binary.dat")
        assert result == data

    @mock_aws
    def test_roundtrip_text(self):
        """UTF-8 text round-trips correctly via pipe/cat."""
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        fs = Boto3S3FileSystem()
        text = "Hello, world! \U0001f30d\nLine 2\n"
        data = text.encode("utf-8")
        fs.pipe_file(f"{BUCKET}/text.txt", data)
        result = fs.cat_file(f"{BUCKET}/text.txt")
        assert result == data
        assert result.decode("utf-8") == text

    @mock_aws
    def test_roundtrip_empty_file(self):
        """Empty file round-trips correctly."""
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        fs = Boto3S3FileSystem()
        fs.pipe_file(f"{BUCKET}/empty.dat", b"")
        result = fs.cat_file(f"{BUCKET}/empty.dat")
        assert result == b""


# --- Test _info ---


class TestInfo:
    """Tests for _info metadata retrieval."""

    @mock_aws
    def test_info_file(self):
        """_info returns correct metadata for a file."""
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        fs = Boto3S3FileSystem()
        fs.pipe_file(f"{BUCKET}/dir/file.txt", b"hello")
        info = fs.info(f"{BUCKET}/dir/file.txt")
        assert info["type"] == "file"
        assert info["size"] == 5
        assert "name" in info

    @mock_aws
    def test_info_directory(self):
        """_info returns directory type for a prefix with children."""
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        fs = Boto3S3FileSystem()
        fs.pipe_file(f"{BUCKET}/dir/file.txt", b"hello")
        info = fs.info(f"{BUCKET}/dir")
        assert info["type"] == "directory"
        assert info["size"] == 0

    @mock_aws
    def test_info_nonexistent_raises(self):
        """_info raises FileNotFoundError for nonexistent paths."""
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        fs = Boto3S3FileSystem()
        with pytest.raises(FileNotFoundError):
            fs.info(f"{BUCKET}/nonexistent/path")


# --- Test _exists ---


class TestExists:
    """Tests for _exists path existence checks."""

    @mock_aws
    def test_exists_file(self):
        """_exists returns True for an existing file."""
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        fs = Boto3S3FileSystem()
        fs.pipe_file(f"{BUCKET}/exists.txt", b"data")
        assert fs.exists(f"{BUCKET}/exists.txt") is True

    @mock_aws
    def test_exists_directory(self):
        """_exists returns True for a non-empty prefix (directory)."""
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        fs = Boto3S3FileSystem()
        fs.pipe_file(f"{BUCKET}/dir/file.txt", b"data")
        assert fs.exists(f"{BUCKET}/dir") is True

    @mock_aws
    def test_exists_nonexistent(self):
        """_exists returns False for non-existent paths."""
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        fs = Boto3S3FileSystem()
        assert fs.exists(f"{BUCKET}/nope") is False


# --- Test _ls ---


class TestLs:
    """Tests for _ls directory listing operations."""

    @mock_aws
    def test_ls_detail_false(self):
        """_ls with detail=False returns list of path strings."""
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        fs = Boto3S3FileSystem()
        fs.pipe_file(f"{BUCKET}/dir/a.txt", b"a")
        fs.pipe_file(f"{BUCKET}/dir/b.txt", b"b")
        result = fs.ls(f"{BUCKET}/dir", detail=False)
        assert sorted(result) == sorted([f"{BUCKET}/dir/a.txt", f"{BUCKET}/dir/b.txt"])

    @mock_aws
    def test_ls_detail_true(self):
        """_ls with detail=True returns list of info dicts."""
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        fs = Boto3S3FileSystem()
        fs.pipe_file(f"{BUCKET}/dir/file.txt", b"hello")
        result = fs.ls(f"{BUCKET}/dir", detail=True)
        assert len(result) == 1
        assert result[0]["name"] == f"{BUCKET}/dir/file.txt"
        assert result[0]["type"] == "file"
        assert result[0]["size"] == 5

    @mock_aws
    def test_ls_with_subdirectories(self):
        """_ls shows both files and subdirectory prefixes."""
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        fs = Boto3S3FileSystem()
        fs.pipe_file(f"{BUCKET}/dir/file.txt", b"data")
        fs.pipe_file(f"{BUCKET}/dir/subdir/nested.txt", b"nested")
        result = fs.ls(f"{BUCKET}/dir", detail=False)
        assert f"{BUCKET}/dir/file.txt" in result
        assert f"{BUCKET}/dir/subdir" in result

    @mock_aws
    def test_ls_empty_prefix_raises(self):
        """_ls raises FileNotFoundError for empty prefix."""
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        fs = Boto3S3FileSystem()
        with pytest.raises(FileNotFoundError):
            fs.ls(f"{BUCKET}/nonexistent")

    @mock_aws
    def test_ls_bare_bucket(self):
        """_ls with bare bucket name lists top-level objects and prefixes."""
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        fs = Boto3S3FileSystem()
        fs.pipe_file(f"{BUCKET}/top.txt", b"top")
        fs.pipe_file(f"{BUCKET}/subdir/nested.txt", b"nested")
        result = fs.ls(BUCKET, detail=False)
        assert f"{BUCKET}/top.txt" in result
        assert f"{BUCKET}/subdir" in result

    @mock_aws
    def test_ls_pagination(self):
        """_ls handles pagination for >1000 objects."""
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        # Create 1005 objects to trigger pagination
        for i in range(1005):
            client.put_object(Bucket=BUCKET, Key=f"many/{i:04d}.txt", Body=b"x")
        fs = Boto3S3FileSystem()
        result = fs.ls(f"{BUCKET}/many", detail=False)
        assert len(result) == 1005


# --- Test _rm_file and recursive _rm ---


class TestRm:
    """Tests for _rm_file and recursive _rm operations."""

    @mock_aws
    def test_rm_file(self):
        """_rm_file deletes a single object."""
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        fs = Boto3S3FileSystem()
        fs.pipe_file(f"{BUCKET}/to_delete.txt", b"delete me")
        assert fs.exists(f"{BUCKET}/to_delete.txt") is True
        fs.rm_file(f"{BUCKET}/to_delete.txt")
        assert fs.exists(f"{BUCKET}/to_delete.txt") is False

    @mock_aws
    def test_rm_file_nonexistent_raises(self):
        """_rm_file raises FileNotFoundError for missing objects."""
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        fs = Boto3S3FileSystem()
        with pytest.raises(FileNotFoundError):
            fs.rm_file(f"{BUCKET}/nope.txt")

    @mock_aws
    def test_rm_recursive(self):
        """_rm with recursive=True deletes all objects under a prefix."""
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        fs = Boto3S3FileSystem()
        fs.pipe_file(f"{BUCKET}/dir/a.txt", b"a")
        fs.pipe_file(f"{BUCKET}/dir/b.txt", b"b")
        fs.pipe_file(f"{BUCKET}/dir/sub/c.txt", b"c")
        fs.rm(f"{BUCKET}/dir", recursive=True)
        assert fs.exists(f"{BUCKET}/dir") is False


# --- Test _put_file and _get_file ---


class TestPutGetFile:
    """Tests for _put_file and _get_file (local to S3 transfers)."""

    @mock_aws
    def test_put_file(self, tmp_path):
        """_put_file uploads a local file to S3."""
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        fs = Boto3S3FileSystem()
        local_file = tmp_path / "upload.txt"
        local_file.write_bytes(b"local content")
        fs.put_file(str(local_file), f"{BUCKET}/uploaded.txt")
        result = fs.cat_file(f"{BUCKET}/uploaded.txt")
        assert result == b"local content"

    @mock_aws
    def test_get_file(self, tmp_path):
        """_get_file downloads an S3 object to a local file."""
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        fs = Boto3S3FileSystem()
        fs.pipe_file(f"{BUCKET}/remote.txt", b"remote content")
        local_file = tmp_path / "download.txt"
        fs.get_file(f"{BUCKET}/remote.txt", str(local_file))
        assert local_file.read_bytes() == b"remote content"


# --- Test _open with various modes ---


class TestOpen:
    """Tests for _open with various file modes."""

    @mock_aws
    def test_open_rb(self):
        """open(mode='rb') reads binary content."""
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        fs = Boto3S3FileSystem()
        fs.pipe_file(f"{BUCKET}/file.bin", b"\x00\x01\x02")
        with fs.open(f"{BUCKET}/file.bin", "rb") as f:
            assert f.read() == b"\x00\x01\x02"

    @mock_aws
    def test_open_wb(self):
        """open(mode='wb') writes binary content."""
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        fs = Boto3S3FileSystem()
        with fs.open(f"{BUCKET}/file.bin", "wb") as f:
            f.write(b"\x03\x04\x05")
        assert fs.cat_file(f"{BUCKET}/file.bin") == b"\x03\x04\x05"

    @mock_aws
    def test_open_r_text(self):
        """open(mode='r') reads text content with UTF-8 encoding."""
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        fs = Boto3S3FileSystem()
        text = "Hello, world!\n"
        fs.pipe_file(f"{BUCKET}/text.txt", text.encode("utf-8"))
        with fs.open(f"{BUCKET}/text.txt", "r") as f:
            assert f.read() == text

    @mock_aws
    def test_open_w_text(self):
        """open(mode='w') writes text content with UTF-8 encoding."""
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        fs = Boto3S3FileSystem()
        text = "Hello, world!\n"
        with fs.open(f"{BUCKET}/text.txt", "w") as f:
            f.write(text)
        result = fs.cat_file(f"{BUCKET}/text.txt")
        assert result.decode("utf-8") == text

    @mock_aws
    def test_open_append_existing(self):
        """open(mode='ab') appends to existing file content."""
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        fs = Boto3S3FileSystem()
        fs.pipe_file(f"{BUCKET}/append.txt", b"first")
        with fs.open(f"{BUCKET}/append.txt", "ab") as f:
            f.write(b"second")
        result = fs.cat_file(f"{BUCKET}/append.txt")
        assert result == b"firstsecond"

    @mock_aws
    def test_open_append_new_file(self):
        """open(mode='ab') creates a new file if it doesn't exist."""
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        fs = Boto3S3FileSystem()
        with fs.open(f"{BUCKET}/new_append.txt", "ab") as f:
            f.write(b"new content")
        result = fs.cat_file(f"{BUCKET}/new_append.txt")
        assert result == b"new content"


# --- Test glob patterns ---


class TestGlob:
    """Tests for glob pattern matching."""

    @mock_aws
    def test_glob_star(self):
        """glob with * matches files in a directory."""
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        fs = Boto3S3FileSystem()
        fs.pipe_file(f"{BUCKET}/data/a.json", b"{}")
        fs.pipe_file(f"{BUCKET}/data/b.json", b"{}")
        fs.pipe_file(f"{BUCKET}/data/c.txt", b"text")
        result = fs.glob(f"{BUCKET}/data/*.json")
        assert sorted(result) == sorted(
            [f"{BUCKET}/data/a.json", f"{BUCKET}/data/b.json"]
        )

    @mock_aws
    def test_glob_recursive(self):
        """glob with ** matches files recursively."""
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        fs = Boto3S3FileSystem()
        fs.pipe_file(f"{BUCKET}/root/a.json", b"{}")
        fs.pipe_file(f"{BUCKET}/root/sub/b.json", b"{}")
        fs.pipe_file(f"{BUCKET}/root/sub/deep/c.json", b"{}")
        result = fs.glob(f"{BUCKET}/root/**/*.json")
        assert len(result) == 3
        assert f"{BUCKET}/root/a.json" in result
        assert f"{BUCKET}/root/sub/b.json" in result
        assert f"{BUCKET}/root/sub/deep/c.json" in result

    @mock_aws
    def test_glob_no_match(self):
        """glob returns empty list when no matches."""
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        fs = Boto3S3FileSystem()
        fs.pipe_file(f"{BUCKET}/data/a.txt", b"text")
        result = fs.glob(f"{BUCKET}/data/*.csv")
        assert result == []


# --- Test UPath integration ---


class TestUPathIntegration:
    """Tests for universal-pathlib UPath integration."""

    @mock_aws
    def test_upath_write_text_read_text(self):
        """UPath read_text/write_text work with Boto3S3FileSystem."""
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        from upath import UPath

        p = UPath(f"s3://{BUCKET}/upath_test.txt")
        p.write_text("hello from upath")
        assert p.read_text() == "hello from upath"

    @mock_aws
    def test_upath_exists(self):
        """UPath.exists() works with Boto3S3FileSystem."""
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        from upath import UPath

        p = UPath(f"s3://{BUCKET}/exists_test.txt")
        assert not p.exists()
        p.write_bytes(b"data")
        assert p.exists()

    @mock_aws
    def test_upath_iterdir(self):
        """UPath.iterdir() lists directory contents."""
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        from upath import UPath

        base = UPath(f"s3://{BUCKET}/iterdir_test")
        (base / "a.txt").write_bytes(b"a")
        (base / "b.txt").write_bytes(b"b")
        names = sorted(p.name for p in base.iterdir())
        assert names == ["a.txt", "b.txt"]


# --- Test async operations ---


class TestAsync:
    """Tests for async operations via pytest-asyncio."""

    @pytest.mark.asyncio
    async def test_async_cat_pipe_roundtrip(self):
        """Async _cat_file/_pipe_file round-trip works."""
        with mock_aws():
            boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
            fs = Boto3S3FileSystem(asynchronous=True)
            await fs._pipe_file(f"{BUCKET}/async.txt", b"async data")
            result = await fs._cat_file(f"{BUCKET}/async.txt")
            assert result == b"async data"

    @pytest.mark.asyncio
    async def test_async_concurrent_operations(self):
        """Multiple async operations can run concurrently."""
        with mock_aws():
            boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
            fs = Boto3S3FileSystem(asynchronous=True)
            # Write multiple files concurrently
            paths = [f"{BUCKET}/concurrent/{i}.txt" for i in range(5)]
            await asyncio.gather(
                *[fs._pipe_file(p, f"data-{i}".encode()) for i, p in enumerate(paths)]
            )
            # Read them back concurrently
            results = await asyncio.gather(*[fs._cat_file(p) for p in paths])
            for i, result in enumerate(results):
                assert result == f"data-{i}".encode()

    @pytest.mark.asyncio
    async def test_async_ls(self):
        """Async _ls works correctly."""
        with mock_aws():
            boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
            fs = Boto3S3FileSystem(asynchronous=True)
            await fs._pipe_file(f"{BUCKET}/async_dir/file1.txt", b"1")
            await fs._pipe_file(f"{BUCKET}/async_dir/file2.txt", b"2")
            result = await fs._ls(f"{BUCKET}/async_dir", detail=False)
            assert sorted(result) == sorted(
                [f"{BUCKET}/async_dir/file1.txt", f"{BUCKET}/async_dir/file2.txt"]
            )

    @pytest.mark.asyncio
    async def test_async_exists(self):
        """Async _exists works correctly."""
        with mock_aws():
            boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
            fs = Boto3S3FileSystem(asynchronous=True)
            assert await fs._exists(f"{BUCKET}/nope") is False
            await fs._pipe_file(f"{BUCKET}/yes.txt", b"yes")
            assert await fs._exists(f"{BUCKET}/yes.txt") is True

    @pytest.mark.asyncio
    async def test_async_info(self):
        """Async _info works correctly."""
        with mock_aws():
            boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
            fs = Boto3S3FileSystem(asynchronous=True)
            await fs._pipe_file(f"{BUCKET}/info.txt", b"12345")
            info = await fs._info(f"{BUCKET}/info.txt")
            assert info["type"] == "file"
            assert info["size"] == 5

    @pytest.mark.asyncio
    async def test_async_rm(self):
        """Async _rm_file works correctly."""
        with mock_aws():
            boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
            fs = Boto3S3FileSystem(asynchronous=True)
            await fs._pipe_file(f"{BUCKET}/rm_async.txt", b"del")
            assert await fs._exists(f"{BUCKET}/rm_async.txt") is True
            await fs._rm_file(f"{BUCKET}/rm_async.txt")
            assert await fs._exists(f"{BUCKET}/rm_async.txt") is False


# --- Test error handling ---


class TestErrorHandling:
    """Tests for FileNotFoundError and PermissionError exceptions."""

    @mock_aws
    def test_cat_file_not_found(self):
        """_cat_file raises FileNotFoundError for missing objects."""
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        fs = Boto3S3FileSystem()
        with pytest.raises(FileNotFoundError):
            fs.cat_file(f"{BUCKET}/missing.txt")

    @mock_aws
    def test_info_not_found(self):
        """_info raises FileNotFoundError for missing paths."""
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        fs = Boto3S3FileSystem()
        with pytest.raises(FileNotFoundError):
            fs.info(f"{BUCKET}/no/such/path")

    @mock_aws
    def test_rm_file_not_found(self):
        """_rm_file raises FileNotFoundError for missing objects."""
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        fs = Boto3S3FileSystem()
        with pytest.raises(FileNotFoundError):
            fs.rm_file(f"{BUCKET}/nope.txt")

    @mock_aws
    def test_permission_error_on_cat(self):
        """_cat_file raises PermissionError on 403."""
        from botocore.exceptions import ClientError

        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        fs = Boto3S3FileSystem()

        def mock_get(**kwargs):
            raise ClientError(
                {"Error": {"Code": "403", "Message": "Access Denied"}},
                "GetObject",
            )

        fs._s3_client.get_object = mock_get
        with pytest.raises(PermissionError):
            fs.cat_file(f"{BUCKET}/forbidden.txt")

    @mock_aws
    def test_permission_error_on_info(self):
        """_info raises PermissionError on AccessDenied."""
        from botocore.exceptions import ClientError

        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        fs = Boto3S3FileSystem()

        def mock_head(**kwargs):
            raise ClientError(
                {"Error": {"Code": "403", "Message": "Access Denied"}},
                "HeadObject",
            )

        fs._s3_client.head_object = mock_head
        with pytest.raises(PermissionError):
            fs.info(f"{BUCKET}/forbidden.txt")


# --- Test registration fallback behavior ---


class TestRegistration:
    """Tests for registration fallback behavior."""

    def test_registration_skips_when_s3fs_available(self):
        """Registration is skipped when s3fs is importable."""
        fake_s3fs = types.ModuleType("s3fs")
        with patch.dict(sys.modules, {"s3fs": fake_s3fs}):
            import llmeter

            # The function should return early without registering
            llmeter._register_s3_filesystem()

    def test_registration_skips_when_obstore_available(self):
        """Registration is skipped when obstore.fsspec is importable."""
        fake_obstore = types.ModuleType("obstore")
        fake_obstore_fsspec = types.ModuleType("obstore.fsspec")
        fake_obstore_fsspec.FsspecStore = object
        with patch.dict(
            sys.modules,
            {
                "obstore": fake_obstore,
                "obstore.fsspec": fake_obstore_fsspec,
                "s3fs": None,  # Simulate s3fs not available (raises ImportError)
            },
        ):
            import llmeter

            llmeter._register_s3_filesystem()

    @mock_aws
    def test_registration_registers_when_no_alternatives(self):
        """Registration registers Boto3S3FileSystem when no alternatives exist."""
        # Ensure s3fs and obstore are NOT importable
        with patch.dict(
            sys.modules,
            {"s3fs": None, "obstore": None, "obstore.fsspec": None},
        ):
            import llmeter

            llmeter._register_s3_filesystem()
            # Verify registration worked by using the filesystem
            boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
            import fsspec

            fs = fsspec.filesystem("s3")
            assert isinstance(fs, Boto3S3FileSystem)

    def test_registration_handles_error_gracefully(self):
        """Registration does not raise if fsspec.register_implementation fails."""
        with patch.dict(
            sys.modules,
            {"s3fs": None, "obstore": None, "obstore.fsspec": None},
        ):
            with patch(
                "fsspec.register_implementation", side_effect=RuntimeError("oops")
            ):
                import llmeter

                # Should not raise - just logs a warning
                llmeter._register_s3_filesystem()
