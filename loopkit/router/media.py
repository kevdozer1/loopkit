"""Images for model calls: one normalized form, hashed for the cache key.

An image may be a numpy ``HxWx3`` uint8 array (also ``HxW`` or ``HxWx4``), a PIL
image, PNG or JPEG bytes, or a path to an image file. Arrays and PIL images are
encoded as PNG, which is lossless: LED colors matter, and JPEG shifts them. PNG and
JPEG bytes and files are sent as they are; any other format PIL can read is
re-encoded as PNG. Each image's sha256 is taken over the bytes that are sent.
"""

from __future__ import annotations

import base64
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"
_PNG_COMPRESS_LEVEL = 6  # fixed so the same pixels give the same bytes and the same hash


@dataclass(frozen=True)
class ImageData:
    data: bytes
    mime: str
    sha256: str
    source: str

    def data_url(self) -> str:
        return f"data:{self.mime};base64," + base64.b64encode(self.data).decode("ascii")

    def __repr__(self) -> str:  # never dump the bytes
        return f"ImageData(mime={self.mime!r}, bytes={len(self.data)}, sha256={self.sha256[:12]}..., source={self.source!r})"


def _sniff(data: bytes) -> str | None:
    if data.startswith(PNG_MAGIC):
        return "image/png"
    if data.startswith(JPEG_MAGIC):
        return "image/jpeg"
    return None


def _png_from_pil(img: Any) -> bytes:
    if img.mode not in ("RGB", "RGBA", "L", "LA"):
        has_alpha = "A" in img.getbands() or img.info.get("transparency") is not None
        img = img.convert("RGBA" if has_alpha else "RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=_PNG_COMPRESS_LEVEL)
    return buf.getvalue()


def _png_from_array(arr: Any) -> bytes:
    import numpy as np
    from PIL import Image

    a = np.asarray(arr)
    if a.dtype != np.uint8:
        raise ValueError(f"image arrays must be uint8 (got {a.dtype}); scale to 0..255 and cast first")
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[:, :, 0]
    if not (a.ndim == 2 or (a.ndim == 3 and a.shape[2] in (3, 4))):
        raise ValueError(f"image arrays must be HxW, HxWx3 or HxWx4 (got shape {a.shape})")
    if a.shape[0] == 0 or a.shape[1] == 0:
        raise ValueError(f"image array is empty (shape {a.shape})")
    return _png_from_pil(Image.fromarray(np.ascontiguousarray(a)))


def _from_bytes(data: bytes, source: str) -> ImageData:
    mime = _sniff(data)
    if mime is None:
        from PIL import Image, UnidentifiedImageError

        try:
            with Image.open(io.BytesIO(data)) as img:
                img.load()
                data = _png_from_pil(img)
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError(f"image {source} is not PNG, JPEG or another format PIL can read") from exc
        mime = "image/png"
    return ImageData(data=data, mime=mime, sha256=hashlib.sha256(data).hexdigest(), source=source)


def normalize_image(image: Any) -> ImageData:
    """One image in any accepted form to :class:`ImageData`."""
    if isinstance(image, ImageData):
        return image
    if isinstance(image, (bytes, bytearray, memoryview)):
        data = bytes(image)
        if not data:
            raise ValueError("image bytes are empty")
        return _from_bytes(data, "bytes")
    if isinstance(image, (str, Path)):
        path = Path(image)
        if not path.is_file():
            raise ValueError(f"image path does not exist: {path}")
        return _from_bytes(path.read_bytes(), "path")
    try:
        from PIL import Image

        if isinstance(image, Image.Image):
            data = _png_from_pil(image)
            return ImageData(data=data, mime="image/png", sha256=hashlib.sha256(data).hexdigest(), source="pil")
    except ImportError:  # pragma: no cover - pillow is a dependency
        pass
    if hasattr(image, "__array__") or type(image).__module__.startswith("numpy"):
        data = _png_from_array(image)
        return ImageData(data=data, mime="image/png", sha256=hashlib.sha256(data).hexdigest(), source="array")
    raise TypeError(f"unsupported image type {type(image).__name__}: pass a uint8 array, a PIL image, "
                    "PNG/JPEG bytes or a path")


def normalize_images(images: list[Any] | tuple[Any, ...] | None) -> list[ImageData]:
    if images is None:
        return []
    if not isinstance(images, (list, tuple)):
        raise TypeError("images must be a list (one entry per image)")
    return [normalize_image(img) for img in images]
