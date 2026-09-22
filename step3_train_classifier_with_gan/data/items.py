"""Dataset item listing: one CXRItem per image."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from common.config import DatasetSpec

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


@dataclass
class CXRItem:
    img_path: Path
    label: int
    site: str
    fx_label_path: Optional[Path] = None
    point_path: Optional[Path] = None


def build_items(spec: DatasetSpec) -> List[CXRItem]:
    """One item per image under <images_root>/{nofx,fx}; fxlabel/maskpoint
    JSONs are attached when present. Item `site` is the dataset name."""
    base = Path(spec.images_root)
    items: List[CXRItem] = []
    for cls_name, label in (("nofx", 0), ("fx", 1)):
        img_dir = base / cls_name
        if not img_dir.exists():
            continue
        for img_path in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS):
            stem = img_path.stem
            fx_label = base / "fxlabel" / f"{stem}.json"
            point = base / "maskpoint" / f"{stem}.json"
            items.append(CXRItem(
                img_path=img_path,
                label=label,
                site=spec.name,
                fx_label_path=fx_label if fx_label.exists() else None,
                point_path=point if point.exists() else None,
            ))
    if not items:
        raise FileNotFoundError(f"[err] no images found under {base}/(fx|nofx)")
    return items
