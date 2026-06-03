# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Async fsspec filesystem for S3 using boto3 directly.

This module provides a self-contained S3 filesystem implementation that avoids
the s3fs/aiobotocore dependency chain which causes version conflicts with boto3.
It implements fsspec's AsyncFileSystem interface, wrapping synchronous boto3 calls
in asyncio.to_thread() for non-blocking execution.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import boto3
import fsspec.spec
from botocore.exceptions import ClientError
from fsspec.asyn import AsyncFileSystem

logger = logging.getLogger(__name__)


def _translate_error(e: ClientError, path: str) -> OSError:
    """Map a boto3 ClientError to the appropriate Python exception."""
    code = e.response["Error"]["Code"]
    if code in ("NoSuchKey", "404", "NoSuchBucket"):
        return FileNotFoundError(path)
    if code in ("403", "AccessDenied"):
        return PermissionError(path)
    return OSError(f"S3 error {code}: {e.response['Error'].get('Message', '')}")


class S3File(fsspec.spec.AbstractBufferedFile):
    """File-like object for S3 backed by in-memory buffer."""

    def _fetch_range(self, start: int, end: int) -> bytes:
        """Download a range of bytes from S3."""
        bucket, key = self.fs._split_path(self.path)
        try:
            response = self.fs._s3_client.get_object(
                Bucket=bucket, Key=key, Range=f"bytes={start}-{end - 1}"
            )
            return response["Body"].read()
        except ClientError as e:
            raise _translate_error(e, self.path) from e

    def _initiate_upload(self) -> None:
        """Initialize upload state. Buffer is already set by AbstractBufferedFile."""
        pass

    def _upload_chunk(self, final: bool = False) -> None:
        """Upload buffered content to S3 via put_object when final=True."""
        if final:
            self.buffer.seek(0)
            bucket, key = self.fs._split_path(self.path)
            data = self.buffer.read()
            try:
                self.fs._s3_client.put_object(Bucket=bucket, Key=key, Body=data)
            except ClientError as e:
                raise _translate_error(e, self.path) from e


class Boto3S3FileSystem(AsyncFileSystem):
    """Async fsspec filesystem for S3 using boto3 directly.

    This implementation wraps synchronous boto3 calls in asyncio.to_thread()
    for non-blocking execution. It avoids the s3fs/aiobotocore dependency chain
    that causes version conflicts with boto3.

    Parameters
    ----------
    session : boto3.Session, optional
        Custom boto3 session for credential management.
    region_name : str, optional
        AWS region for the S3 client.
    endpoint_url : str, optional
        Custom endpoint URL for S3-compatible services (e.g., MinIO, LocalStack).
    """

    protocol = ("s3", "s3a")
    async_impl = True
    use_listings_cache = False

    def __init__(
        self,
        session: Any | None = None,
        region_name: str | None = None,
        endpoint_url: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._session = session
        self._region_name = region_name
        self._endpoint_url = endpoint_url
        self._client: Any | None = None

    @property
    def _s3_client(self):
        """Lazy boto3 S3 client creation."""
        if self._client is None:
            kwargs: dict[str, Any] = {}
            if self._region_name:
                kwargs["region_name"] = self._region_name
            if self._endpoint_url:
                kwargs["endpoint_url"] = self._endpoint_url

            if self._session is not None:
                self._client = self._session.client("s3", **kwargs)
            else:
                self._client = boto3.client("s3", **kwargs)
        return self._client

    @staticmethod
    def _strip_protocol(path: str) -> str:
        """Strip s3:// or s3a:// prefix and normalize."""
        if isinstance(path, list):
            return [Boto3S3FileSystem._strip_protocol(p) for p in path]
        for proto in ("s3://", "s3a://"):
            if path.startswith(proto):
                path = path[len(proto) :]
        return path.strip("/")

    def _split_path(self, path: str) -> tuple[str, str]:
        """Split a normalized path into (bucket, key)."""
        path = self._strip_protocol(path)
        if "/" in path:
            bucket, key = path.split("/", 1)
        else:
            bucket, key = path, ""
        return bucket, key

    # --- Async core methods ---

    async def _cat_file(
        self, path: str, start: int | None = None, end: int | None = None, **kwargs: Any
    ) -> bytes:
        """Read file contents from S3."""
        bucket, key = self._split_path(path)

        def _do_get():
            get_kwargs: dict[str, Any] = {"Bucket": bucket, "Key": key}
            if start is not None or end is not None:
                s = start or 0
                e = f"{end - 1}" if end else ""
                get_kwargs["Range"] = f"bytes={s}-{e}"
            try:
                response = self._s3_client.get_object(**get_kwargs)
                return response["Body"].read()
            except ClientError as e:
                raise _translate_error(e, path) from e

        return await asyncio.to_thread(_do_get)

    async def _pipe_file(self, path: str, value: bytes, **kwargs: Any) -> None:
        """Write bytes content to S3."""
        bucket, key = self._split_path(path)

        def _do_put():
            try:
                self._s3_client.put_object(Bucket=bucket, Key=key, Body=value)
            except ClientError as e:
                raise _translate_error(e, path) from e

        await asyncio.to_thread(_do_put)

    async def _info(self, path: str, **kwargs: Any) -> dict[str, Any]:
        """Get metadata for an S3 path."""
        bucket, key = self._split_path(path)

        def _do_info():
            # Try as file first (head_object)
            if key:
                try:
                    response = self._s3_client.head_object(Bucket=bucket, Key=key)
                    return {
                        "name": path,
                        "type": "file",
                        "size": response["ContentLength"],
                        "LastModified": response.get("LastModified"),
                        "ETag": response.get("ETag"),
                    }
                except ClientError as e:
                    err = _translate_error(e, path)
                    if not isinstance(err, FileNotFoundError):
                        raise err from e

            # Try as directory (prefix check)
            prefix = f"{key}/" if key else ""
            response = self._s3_client.list_objects_v2(
                Bucket=bucket, Prefix=prefix, MaxKeys=1
            )
            if response.get("KeyCount", 0) > 0:
                return {
                    "name": path,
                    "type": "directory",
                    "size": 0,
                }

            raise FileNotFoundError(path)

        return await asyncio.to_thread(_do_info)

    async def _exists(self, path: str, **kwargs: Any) -> bool:
        """Check if an S3 path exists (as object or non-empty prefix)."""
        try:
            await self._info(path)
            return True
        except FileNotFoundError:
            return False

    async def _ls(self, path: str, detail: bool = False, **kwargs: Any) -> list:
        """List objects and prefixes under an S3 path."""
        bucket, key = self._split_path(path)

        def _do_ls():
            prefix = f"{key}/" if key else ""
            results: list[dict[str, Any]] = []
            continuation_token = None

            while True:
                list_kwargs: dict[str, Any] = {
                    "Bucket": bucket,
                    "Prefix": prefix,
                    "Delimiter": "/",
                }
                if continuation_token:
                    list_kwargs["ContinuationToken"] = continuation_token

                response = self._s3_client.list_objects_v2(**list_kwargs)

                # Common prefixes (directories)
                for cp in response.get("CommonPrefixes", []):
                    name = f"{bucket}/{cp['Prefix'].rstrip('/')}"
                    results.append({"name": name, "type": "directory", "size": 0})

                # Objects (files)
                for obj in response.get("Contents", []):
                    obj_key = obj["Key"]
                    if obj_key == prefix:
                        continue
                    name = f"{bucket}/{obj_key}"
                    results.append(
                        {
                            "name": name,
                            "type": "file",
                            "size": obj["Size"],
                            "LastModified": obj.get("LastModified"),
                            "ETag": obj.get("ETag"),
                        }
                    )

                if response.get("IsTruncated"):
                    continuation_token = response["NextContinuationToken"]
                else:
                    break

            if not results:
                raise FileNotFoundError(path)

            return results

        entries = await asyncio.to_thread(_do_ls)

        if detail:
            return entries
        return [entry["name"] for entry in entries]

    async def _mkdir(
        self, path: str, create_parents: bool = True, **kwargs: Any
    ) -> None:
        """No-op: S3 does not have real directories."""
        pass

    async def _rm_file(self, path: str, **kwargs: Any) -> None:
        """Delete a single object from S3."""
        bucket, key = self._split_path(path)
        if not key:
            raise ValueError(f"Cannot delete a bare bucket: {path}")

        def _do_rm():
            # delete_object is idempotent on S3 — no existence check needed
            self._s3_client.delete_object(Bucket=bucket, Key=key)

        await asyncio.to_thread(_do_rm)

    async def _rm(self, path, recursive=False, batch_size=None, **kwargs):
        """Delete files, skipping directory placeholders that don't exist as objects."""
        from fsspec.asyn import _run_coros_in_chunks

        batch_size = batch_size or self.batch_size
        paths = await self._expand_path(path, recursive=recursive)

        async def _safe_rm(p):
            try:
                await self._rm_file(p, **kwargs)
            except (FileNotFoundError, ValueError):
                # Expanded paths may include directory prefixes or bare buckets
                pass

        return await _run_coros_in_chunks(
            [_safe_rm(p) for p in reversed(paths)],
            batch_size=batch_size,
            nofiles=True,
        )

    async def _cp_file(self, path1: str, path2: str, **kwargs: Any) -> None:
        """Copy an S3 object to another location."""
        bucket1, key1 = self._split_path(path1)
        bucket2, key2 = self._split_path(path2)

        def _do_copy():
            try:
                self._s3_client.copy_object(
                    Bucket=bucket2,
                    Key=key2,
                    CopySource={"Bucket": bucket1, "Key": key1},
                )
            except ClientError as e:
                raise _translate_error(e, path1) from e

        await asyncio.to_thread(_do_copy)

    async def _put_file(self, lpath: str, rpath: str, **kwargs: Any) -> None:
        """Upload a local file to S3."""
        bucket, key = self._split_path(rpath)

        def _do_upload():
            self._s3_client.upload_file(lpath, bucket, key)

        await asyncio.to_thread(_do_upload)

    async def _get_file(self, rpath: str, lpath: str, **kwargs: Any) -> None:
        """Download an S3 object to a local file."""
        bucket, key = self._split_path(rpath)

        def _do_download():
            try:
                self._s3_client.download_file(bucket, key, lpath)
            except ClientError as e:
                raise _translate_error(e, rpath) from e

        await asyncio.to_thread(_do_download)

    def _open(
        self,
        path: str,
        mode: str = "rb",
        block_size: int | None = None,
        autocommit: bool = True,
        cache_options: dict | None = None,
        **kwargs: Any,
    ) -> S3File:
        """Return a file-like object for S3."""
        path = self._strip_protocol(path)

        if "a" in mode:
            # Append mode: pre-fetch existing content (non-atomic read-modify-write)
            try:
                existing = self.cat_file(path)
            except FileNotFoundError:
                existing = b""
            f = S3File(
                self,
                path,
                mode="wb",
                block_size=block_size,
                autocommit=autocommit,
                cache_options=cache_options,
                **kwargs,
            )
            f.write(existing)
            return f

        return S3File(
            self,
            path,
            mode=mode,
            block_size=block_size,
            autocommit=autocommit,
            cache_options=cache_options,
            **kwargs,
        )
