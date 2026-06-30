"""TableRecPredictor — RT-DETRv2 ONNX table-structure detector (CPU).

Detects Row/Col boxes on a cropped table image and derives cells as
row × column intersections, returning a TableResult. Input images should be
table crops.
"""

from __future__ import annotations

from typing import List, Optional

from PIL import Image

from surya.common.rfdetr_torch import load_detector
from surya.common.rtdetr_onnx import resolve_model_dir
from surya.settings import settings
from surya.table_rec.schema import TableCell, TableCol, TableResult, TableRow


def _poly(b):
    x0, y0, x1, y1 = b
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _intersect(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    return None if (x1 <= x0 or y1 <= y0) else (x0, y0, x1, y1)


class TableRecPredictor:
    def __init__(
        self, checkpoint: Optional[str] = None, num_threads: Optional[int] = None
    ):
        model_dir = resolve_model_dir(
            checkpoint or settings.FAST_TABLE_MODEL_CHECKPOINT
        )
        self.model = load_detector(
            model_dir, num_threads=num_threads, device=settings.FAST_DETECTOR_DEVICE
        )
        self._disable_tqdm = settings.DISABLE_TQDM

    def to(self, *args, **kwargs):
        return

    def __call__(
        self,
        images: List[Image.Image],
        threshold: Optional[float] = None,
        batch_size: Optional[int] = None,
    ) -> List[TableResult]:
        if not images:
            return []
        threshold = (
            settings.FAST_TABLE_CONFIDENCE_THRESHOLD if threshold is None else threshold
        )
        batch_size = batch_size or settings.FAST_TABLE_BATCH_SIZE or 8
        detections = self.model.detect(
            images, threshold=threshold, batch_size=batch_size
        )

        results: List[TableResult] = []
        for image, dets in zip(images, detections):
            # rows sorted top->bottom, cols left->right (stable ids)
            row_dets = sorted(
                [d for d in dets if d["label"] == "Row"], key=lambda d: d["bbox"][1]
            )
            col_dets = sorted(
                [d for d in dets if d["label"] == "Col"], key=lambda d: d["bbox"][0]
            )
            rows = [
                TableRow(polygon=_poly(d["bbox"]), row_id=i, confidence=d["score"])
                for i, d in enumerate(row_dets)
            ]
            cols = [
                TableCol(polygon=_poly(d["bbox"]), col_id=i, confidence=d["score"])
                for i, d in enumerate(col_dets)
            ]
            cells: List[TableCell] = []
            cid = 0
            for row in rows:
                for col in cols:
                    inter = _intersect(row.bbox, col.bbox)
                    if inter is None:
                        continue
                    cells.append(
                        TableCell(
                            polygon=_poly(inter),
                            row_id=row.row_id,
                            col_id=col.col_id,
                            cell_id=cid,
                        )
                    )
                    cid += 1
            results.append(
                TableResult(
                    rows=rows,
                    cols=cols,
                    cells=cells,
                    image_bbox=[0.0, 0.0, float(image.width), float(image.height)],
                )
            )
        return results
