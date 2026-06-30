"""FastLayoutPredictor — RT-DETRv2 ONNX page-layout detector (CPU).

Drop-in alternative to surya.layout.LayoutPredictor: same LayoutResult/LayoutBox output, but
a ~20M-param CPU ONNX detector instead of the VLM. Labels are canonicalized through the same
LAYOUT_PRED_RELABEL map the VLM layout model uses, so downstream consumers (marker) are unchanged.
"""

from __future__ import annotations

from typing import List, Optional

from PIL import Image

from surya.common.rfdetr_torch import load_detector
from surya.common.order.predictor import load_order_predictor
from surya.common.rtdetr_onnx import resolve_model_dir
from surya.layout.label import LAYOUT_PRED_RELABEL
from surya.layout.schema import LayoutBox, LayoutResult
from surya.logging import get_logger
from surya.settings import settings

logger = get_logger()


def _poly(b):
    x0, y0, x1, y1 = b
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


class FastLayoutPredictor:
    def __init__(
        self, checkpoint: Optional[str] = None, num_threads: Optional[int] = None
    ):
        model_dir = resolve_model_dir(
            checkpoint or settings.FAST_LAYOUT_MODEL_CHECKPOINT
        )
        self.model = load_detector(
            model_dir, num_threads=num_threads, device=settings.FAST_DETECTOR_DEVICE
        )
        # Learned reading-order head (cross-attends to the detector's encoder feature map).
        # Always runs alongside layout when a checkpoint is configured/available; falls back to
        # a raster sort otherwise.
        self.order = load_order_predictor(device=settings.FAST_DETECTOR_DEVICE or "cpu")
        if self.order is None:
            logger.warning(
                "Reading-order model not available; falling back to raster sort "
                "(top-to-bottom, left-to-right) for all pages."
            )
        self._disable_tqdm = settings.DISABLE_TQDM

    def to(
        self, *args, **kwargs
    ):  # API parity with other predictors (no-op on CPU ONNX)
        return

    def __call__(
        self,
        images: List[Image.Image],
        threshold: Optional[float] = None,
        batch_size: Optional[int] = None,
    ) -> List[LayoutResult]:
        if not images:
            return []
        threshold = (
            settings.FAST_LAYOUT_CONFIDENCE_THRESHOLD
            if threshold is None
            else threshold
        )
        batch_size = batch_size or settings.FAST_LAYOUT_BATCH_SIZE or 8
        want_feats = self.order is not None
        detections = self.model.detect(
            images,
            threshold=threshold,
            batch_size=batch_size,
            return_features=want_feats,
        )

        results: List[LayoutResult] = []
        for image, dets in zip(images, detections):
            # Reading order: the learned AR head (cross-attends to the encoder feature map) when
            # available, else a top-to-bottom / left-to-right raster sort.
            feats = getattr(dets, "features", None)
            if self.order is not None and feats is not None and dets:
                positions = self.order.order_page(
                    feats,
                    [d["bbox"] for d in dets],
                    [d["label"] for d in dets],
                    image.width,
                    image.height,
                )
            else:
                # Order model loaded but no feature map came back — it should have run
                # but didn't. Surface this; the "model never loaded" case is logged once at init.
                if self.order is not None and feats is None and dets:
                    logger.warning(
                        "Reading-order model loaded but detector returned no feature map; "
                        "falling back to raster sort for this page."
                    )
                order = sorted(
                    range(len(dets)),
                    key=lambda i: (dets[i]["bbox"][1], dets[i]["bbox"][0]),
                )
                positions = [0] * len(dets)
                for rank, i in enumerate(order):
                    positions[i] = rank
            boxes = []
            for d, pos in zip(dets, positions):
                raw = d["label"]
                boxes.append(
                    LayoutBox(
                        polygon=_poly(d["bbox"]),
                        label=LAYOUT_PRED_RELABEL.get(raw, raw),
                        raw_label=raw,
                        position=pos,
                        confidence=d["score"],
                    )
                )
            boxes.sort(key=lambda b: b.position)
            results.append(
                LayoutResult(
                    bboxes=boxes,
                    image_bbox=[0.0, 0.0, float(image.width), float(image.height)],
                )
            )
        return results
