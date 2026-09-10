"""One frozen, ordered product reference from several views of one product."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageOps

MANIFEST = "work/global-reference-sheet/user-reference-views.json"
REFERENCE = "work/global-reference-sheet/user-reference.png"
CELL_SIZE = 768
LABEL_HEIGHT = 32


def _trim_white_border(image: Image.Image, alpha: Image.Image) -> tuple[Image.Image, tuple[int,int,int,int]]:
    """Crop only supported outer background; this is not object segmentation.

    Mostly white/low-contrast RGB images may be white products with a dark
    logo. Keep those whole rather than declaring the logo the entire product.
    Exact white-on-white boundaries cannot be recovered from RGB alone.
    """
    full = (0,0,image.width,image.height)
    alpha_range = alpha.getextrema()
    if alpha_range[0] < 255:
        # Original alpha carries support even for a white product. A fully
        # transparent upload has no reliable subject, so keep its dimensions.
        bbox = alpha.getbbox()
        if bbox is None:
            return image, full
    else:
        corners = [(0,0),(image.width-1,0),(0,image.height-1),(image.width-1,image.height-1)]
        background = image.getpixel(corners[0])
        if min(background) < 250 or any(image.getpixel(point) != background for point in corners):
            return image, full
        # A small contrast feature must not turn into a crop of a white
        # product's logo. Ambiguous sparse subjects simply use the full image.
        contrast = image.convert("L").point(lambda value: 255 if value < 220 else 0)
        if contrast.histogram()[255] < image.width*image.height*.20:
            return image, full
        # Keep every visible difference, including faint white-product edges;
        # do not threshold away near-white subject pixels or internal whites.
        bbox = ImageChops.difference(image,Image.new("RGB",image.size,background)).getbbox()
        if bbox is None:
            return image, full
    padding = max(2,min(image.size)//100)
    crop = (max(0,bbox[0]-padding),max(0,bbox[1]-padding),
            min(image.width,bbox[2]+padding),min(image.height,bbox[3]+padding))
    return image.crop(crop), crop


def _receipt(project: Path, path: Path) -> dict:
    root = project.resolve(strict=True)
    resolved = path.resolve(strict=True)
    if path.is_symlink() or root not in resolved.parents or not resolved.is_file():
        raise ValueError("product view must be a project-owned file")
    payload = path.read_bytes()
    return {"path": path.relative_to(project).as_posix(),
            "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


def frozen_product_views(project: Path) -> dict | None:
    path = project / MANIFEST
    if not path.exists():
        return None
    if path.is_symlink():
        raise ValueError("product view manifest must be project-owned")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("version") != 1:
        raise ValueError("invalid product view manifest")
    sources = value.get("sources")
    if not isinstance(sources, list) or not 2 <= len(sources) <= 4:
        raise ValueError("invalid product view count")
    expected_paths = ["inputs/replacement-image"] + [f"inputs/replacement-image-{i}" for i in range(2,len(sources)+1)]
    for receipt, expected in zip(sources, expected_paths):
        if not isinstance(receipt, dict) or receipt.get("path") != expected or receipt != _receipt(project, project / expected):
            raise ValueError("frozen product view source changed")
    if value.get("output") != _receipt(project, project / REFERENCE):
        raise ValueError("frozen product view composite changed")
    if value.get("view_count") != len(sources) or value.get("interpretation") != "one_product_multiple_views":
        raise ValueError("invalid product view interpretation")
    return value


def compose_product_views(project: Path, sources: list[Path]) -> Path:
    if not 2 <= len(sources) <= 4:
        raise ValueError("product composite requires 2 to 4 ordered views")
    expected_paths = [project / "inputs/replacement-image"] + [project / f"inputs/replacement-image-{i}" for i in range(2,len(sources)+1)]
    if sources != expected_paths:
        raise ValueError("product view order must match stored upload order")
    source_receipts = [_receipt(project, source) for source in sources]
    existing = frozen_product_views(project)
    if existing is not None:
        if existing["sources"] != source_receipts:
            raise ValueError("frozen product view order changed")
        return project / REFERENCE
    # H3 accepts reference aspect ratios only in [0.4, 2.5]. Three
    # horizontal cells were 3:1; use two columns with one unlabelled blank
    # cell for three views. Frozen composites above are deliberately reused.
    columns = 2
    rows = 2 if len(sources) >= 3 else 1
    canvas = Image.new("RGB", (columns*CELL_SIZE, rows*CELL_SIZE), "white")
    drawing = ImageDraw.Draw(canvas)
    views = []
    for index, source in enumerate(sources):
        with Image.open(source) as image:
            image = ImageOps.exif_transpose(image).convert("RGBA")
            # Flatten alpha onto white, then trim only supported outer padding.
            # Preserve interior white details and unsupported/ambiguous borders.
            background = Image.new("RGBA", image.size, "white")
            background.alpha_composite(image)
            content, source_crop = _trim_white_border(background.convert("RGB"),image.getchannel("A"))
            source_size = list(image.size)
            bounds = (CELL_SIZE-24,CELL_SIZE-LABEL_HEIGHT-24)
            try:
                fitted = ImageOps.contain(content,bounds)
            except ValueError:
                # Pillow's existing thumbnail handles subpixel short edges;
                # no custom scaling formula or new resize tool is required.
                fitted = content.copy()
                fitted.thumbnail(bounds,resample=Image.Resampling.BICUBIC)
        x, y = (index % columns)*CELL_SIZE, (index // columns)*CELL_SIZE
        left, top = x+(CELL_SIZE-fitted.width)//2, y+LABEL_HEIGHT+(CELL_SIZE-LABEL_HEIGHT-fitted.height)//2
        canvas.paste(fitted, (left, top))
        drawing.text((x+12,y+10), f"View {index+1}", fill="black")
        views.append({"view": index+1, "source": source_receipts[index]["path"],
                      "cell": [x,y,CELL_SIZE,CELL_SIZE], "image_bbox": [left,top,fitted.width,fitted.height],
                      "source_size": source_size, "source_crop_bbox": list(source_crop)})
    destination = project / REFERENCE
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(".user-reference-views.png.tmp")
    manifest_path = project / MANIFEST
    manifest_temporary = manifest_path.with_suffix(".json.tmp")
    try:
        canvas.save(temporary, format="PNG")
        temporary.chmod(0o600)
        os.replace(temporary, destination)
        manifest = {"version": 1, "view_count": len(sources),
                    "interpretation": "one_product_multiple_views", "sources": source_receipts,
                    "layout": {"columns": columns, "rows": rows, "views": views},
                    "output": _receipt(project, destination)}
        manifest_temporary.write_text(json.dumps(manifest, sort_keys=True, ensure_ascii=False)+"\n",encoding="utf-8")
        manifest_temporary.chmod(0o600)
        os.replace(manifest_temporary, manifest_path)
    finally:
        temporary.unlink(missing_ok=True)
        manifest_temporary.unlink(missing_ok=True)
    return destination


def view_instruction(count: int, image_name: str) -> str:
    return (
        f"{image_name} is one composite containing {count} uploaded views of the SAME single product, "
        "numbered View 1 onward in upload order. Treat all cells as complementary evidence of one "
        "product, never as different products or additional replacement targets. Use only the visible "
        "details; do not invent missing views or assign unsupported front/side/back labels. "
        "Multiple source appearances may map to this same product. The grid, labels and background "
        "are reference layout only and must not appear in the generated scene."
    )
