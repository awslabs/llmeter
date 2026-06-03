from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "0.0.0"


def _register_s3_filesystem():
    """Register boto3 S3 backend if no other S3 backend is available."""
    try:
        import s3fs  # noqa: F401

        return  # s3fs is installed, defer to it
    except ImportError:
        pass

    try:
        from obstore.fsspec import FsspecStore  # noqa: F401

        return  # obstore is installed, defer to it
    except ImportError:
        pass

    try:
        import fsspec

        fsspec.register_implementation(
            "s3", "llmeter.s3fs.Boto3S3FileSystem", clobber=True
        )
        fsspec.register_implementation(
            "s3a", "llmeter.s3fs.Boto3S3FileSystem", clobber=True
        )
    except Exception:
        import logging

        logging.getLogger(__name__).warning(
            "Failed to register llmeter S3 filesystem backend", exc_info=True
        )


_register_s3_filesystem()
