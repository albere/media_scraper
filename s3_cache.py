"""Async S3-backed cache for per-month corpus CSV files.

Usage::

    async with S3Cache() as cache:
        if await cache.exists(outlet, year, month):
            ...  # already done — skip
        await cache.upload(local_path, outlet, year, month)
        data = await cache.download(local_path, outlet, year, month)

When no bucket is configured (``S3_BUCKET`` env var or the ``bucket``
constructor argument), the instance operates in *pass-through / no-cache*
mode: :meth:`exists` always returns ``False`` and :meth:`upload` /
:meth:`download` are silent no-ops.  This ensures the scrapers work
unchanged when no S3 bucket is set up.
"""

import logging
import os
from pathlib import Path

_log = logging.getLogger(__name__)


class S3Cache:
    """Async S3-backed cache that stores one CSV per ``(outlet, year, month)``.

    Key format: ``{outlet_slug}/{year}-{month:02d}.csv``

    Parameters
    ----------
    bucket:
        S3 bucket name.  Falls back to the ``S3_BUCKET`` environment variable
        when *None*.  When neither is set the instance is disabled and all
        operations become no-ops.
    """

    def __init__(self, bucket: str | None = None) -> None:
        self._bucket = bucket or os.environ.get("S3_BUCKET")
        self._enabled = bool(self._bucket)
        self._client = None
        self._client_ctx = None

        if not self._enabled:
            _log.warning(
                "S3Cache: no bucket configured (set S3_BUCKET env var or pass "
                "--s3-bucket) — operating in pass-through/no-cache mode"
            )

    # ── Async context manager ────────────────────────────────────────────

    async def __aenter__(self) -> "S3Cache":
        if self._enabled:
            try:
                import aiobotocore.session  # lazy import — optional dependency

                session = aiobotocore.session.get_session()
                self._client_ctx = session.create_client("s3")
                self._client = await self._client_ctx.__aenter__()
            except Exception as exc:
                _log.warning(
                    f"S3Cache: could not create S3 client ({exc}) "
                    "— operating in pass-through/no-cache mode"
                )
                self._enabled = False
                self._client_ctx = None
                self._client = None
        return self

    async def __aexit__(self, *args) -> None:
        if self._client_ctx is not None:
            await self._client_ctx.__aexit__(*args)
            self._client_ctx = None
            self._client = None

    # ── Internal helpers ─────────────────────────────────────────────────

    @staticmethod
    def _slug(outlet: str) -> str:
        return outlet.lower().replace(" ", "_")

    def _key(self, outlet: str, year: int, month: int) -> str:
        return f"{self._slug(outlet)}/{year}-{month:02d}.csv"

    @staticmethod
    def _error_code(exc: Exception) -> str:
        """Extract the S3 error code from a botocore ClientError, or ''."""
        try:
            return exc.response["Error"]["Code"]  # type: ignore[attr-defined]
        except (AttributeError, KeyError, TypeError):
            return ""

    # ── Public API ───────────────────────────────────────────────────────

    async def exists(self, outlet: str, year: int, month: int) -> bool:
        """Return ``True`` if the month CSV is already present in S3."""
        if not self._enabled:
            return False
        key = self._key(outlet, year, month)
        try:
            await self._client.head_object(Bucket=self._bucket, Key=key)
            _log.info(f"S3 cache hit: s3://{self._bucket}/{key}")
            return True
        except Exception as exc:
            if self._error_code(exc) in ("404", "NoSuchKey"):
                return False
            _log.warning(f"S3Cache.exists({key}): unexpected error — {exc}")
            return False

    async def upload(
        self, local_path: "str | Path", outlet: str, year: int, month: int
    ) -> None:
        """Upload a completed month CSV to S3."""
        if not self._enabled:
            return
        key = self._key(outlet, year, month)
        _log.info(f"S3Cache: uploading {local_path} → s3://{self._bucket}/{key}")
        body = Path(local_path).read_bytes()
        await self._client.put_object(Bucket=self._bucket, Key=key, Body=body)

    async def download(
        self, local_path: "str | Path", outlet: str, year: int, month: int
    ) -> None:
        """Download a cached month CSV from S3 to *local_path*."""
        if not self._enabled:
            return
        key = self._key(outlet, year, month)
        _log.info(f"S3Cache: downloading s3://{self._bucket}/{key} → {local_path}")
        response = await self._client.get_object(Bucket=self._bucket, Key=key)
        async with response["Body"] as stream:
            data = await stream.read()
        Path(local_path).write_bytes(data)
