"""
Instance Segmentation Visualizer

Draws bounding boxes with semi-transparent masks overlaid on the image.
Each instance gets a color that is stable across frames: a lightweight IoU
tracker assigns a persistent ``track_id`` to every box, and the color is chosen
by ``track_id`` (not by detection order). This keeps the same object the same
color for both its box and mask across frames. Used by YOLOv8-seg, YOLOv5-seg,
YOLO26-seg, YOLACT, FastSAM.
"""

import numpy as np
import cv2
from typing import List

from ..base import IVisualizer
from ..trackers import IoUTracker


def _generate_instance_palette(count: int = 30) -> List[tuple]:
    """Pastel + neon color palette for stable per-track instance colors.

    Hues step by the golden angle so that neighboring palette entries — and
    any wraparound via ``track_id % len(palette)`` once more than ``count``
    objects have been tracked — land far apart on the color wheel instead of
    drifting through adjacent hues. Alternating saturation between a muted
    "pastel" tone and a punchy "neon" tone (both at full value) doubles the
    effective distinctiveness for the same hue spacing.
    """
    golden_angle = 137.508  # degrees; wrapped into OpenCV's 0-179 hue range
    hues = (np.arange(count) * golden_angle) % 180
    hsv = np.zeros((count, 1, 3), dtype=np.uint8)
    hsv[:, 0, 0] = hues.astype(np.uint8)
    hsv[:, 0, 1] = np.where(np.arange(count) % 2 == 0, 235, 110).astype(np.uint8)
    hsv[:, 0, 2] = 255
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return [tuple(int(c) for c in px) for px in bgr[:, 0, :]]


# Pastel + neon colors for instance segmentation (see _generate_instance_palette)
INSTANCE_COLORS = _generate_instance_palette(30)


class InstanceSegVisualizer(IVisualizer):
    """Visualizer for instance segmentation with mask overlay.

    Args:
        label_set: named label set (e.g. 'coco80') when ``labels`` is not given.
        labels: explicit class-name list; overrides ``label_set``.
        show_boxes: draw bounding boxes + labels (False → mask-only, e.g. FastSAM).
        enable_tracking: when True (default) color is bound to a per-object
            ``track_id`` from an IoU tracker so colors stay stable across
            frames; when False, color falls back to detection index.
        iou_threshold / max_age: IoU tracker tuning (see IoUTracker).
        hold_frames: when a tracked object's detection confidence dips below
            the score threshold for a frame or two (common for small/distant/
            partially-occluded instances), its mask would otherwise vanish and
            reappear — a visible flicker. Instead its last-seen mask is redrawn
            for up to this many consecutive missed frames before the track is
            dropped. 0 disables holding. No effect when ``enable_tracking`` is
            False (holding needs a stable ``track_id`` to key off of).
    """

    def __init__(self, label_set: str = 'coco80', labels: List[str] = None,
                 show_boxes: bool = True, enable_tracking: bool = True,
                 iou_threshold: float = 0.3, max_age: int = 30,
                 hold_frames: int = 2):
        self.label_set = label_set
        self.labels = labels if labels is not None else self._load_labels(label_set)
        self.color_palette = INSTANCE_COLORS
        self.show_boxes = show_boxes
        self.enable_tracking = enable_tracking
        self.tracker = IoUTracker(iou_threshold=iou_threshold,
                                  max_age=max_age) if enable_tracking else None
        self.hold_frames = hold_frames if enable_tracking else 0
        # track_id -> {"mask", "box", "color", "missed"} for the hold-last-mask
        # flicker fix.
        self._held = {}

    def _load_labels(self, label_set: str) -> List[str]:
        if label_set == 'coco80':
            return [
                "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
                "truck", "boat", "traffic light", "fire hydrant", "stop sign",
                "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
                "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
                "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
                "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
                "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
                "couch", "potted plant", "bed", "dining table", "toilet", "tv",
                "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
                "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
                "scissors", "teddy bear", "hair drier", "toothbrush"
            ]
        return []

    def _assign_track_ids(self, results: List) -> None:
        """Run the tracker over the current boxes and stamp results in place.

        Color is later chosen from ``track_id`` so the same object keeps its
        color across frames regardless of detection order.
        """
        if self.tracker is None:
            return
        boxes = [r.box for r in results]
        track_ids = self.tracker.update(boxes)
        for r, tid in zip(results, track_ids):
            r.track_id = tid

    def _color_for(self, result, index: int):
        """Pick a palette color: by track_id when tracking, else by index."""
        key = result.track_id if (self.enable_tracking
                                   and getattr(result, 'track_id', -1) >= 0) else index
        return INSTANCE_COLORS[key % len(INSTANCE_COLORS)]

    @staticmethod
    def _draw_mask(overlay: np.ndarray, mask, box, color,
                    img_w: int, img_h: int) -> None:
        """Paint one instance's mask into ``overlay`` within its bbox ROI.

        The postprocessor zeroes each instance mask outside its bbox, so
        restrict the (per-instance, full-frame) boolean index + write to the
        bbox ROI — this is output-identical but avoids allocating/scanning a
        full H*W array per instance. A small margin absorbs sub-pixel spread
        from mask resizing.
        """
        if mask is None or mask.size == 0 or mask.shape[:2] != (img_h, img_w):
            return
        x1, y1, x2, y2 = [int(v) for v in box]
        m = 2
        rx1, ry1 = max(0, x1 - m), max(0, y1 - m)
        rx2, ry2 = min(img_w, x2 + m), min(img_h, y2 + m)
        if rx2 > rx1 and ry2 > ry1:
            roi = mask[ry1:ry2, rx1:rx2]
            overlay[ry1:ry2, rx1:rx2][roi > 0] = color

    def visualize(self, image: np.ndarray, results: List) -> np.ndarray:
        output = image.copy()
        overlay = image.copy()

        # Assign stable track ids for the current frame before drawing so that
        # both the mask and the box of one object use the same color.
        self._assign_track_ids(results)

        # For class-agnostic models (1 class), sort by area descending so large
        # segments are drawn first and smaller ones overlay on top — cleaner look.
        draw_order = list(range(len(results)))
        if len(self.labels) == 1 and len(results) > 1:
            draw_order.sort(key=lambda i: -((
                results[i].box[2] - results[i].box[0]) * (
                results[i].box[3] - results[i].box[1])))

        img_h, img_w = image.shape[:2]
        seen_track_ids = set()
        for i in draw_order:
            r = results[i]
            color = self._color_for(r, i)
            x1, y1, x2, y2 = [int(v) for v in r.box]
            mask = r.mask if hasattr(r, 'mask') else None

            self._draw_mask(overlay, mask, r.box, color, img_w, img_h)

            # Remember this track's mask so a brief drop in detection
            # confidence next frame can redraw it instead of flickering away.
            tid = getattr(r, 'track_id', -1)
            if self.hold_frames > 0 and tid >= 0 and mask is not None and mask.size > 0:
                self._held[tid] = {"mask": mask, "box": r.box,
                                    "color": color, "missed": 0}
                seen_track_ids.add(tid)

            # Draw bounding box and label (skip for mask-only mode)
            if self.show_boxes:
                cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

                # Label
                label = f"{r.class_id}"
                if self.labels and r.class_id < len(self.labels):
                    label = self.labels[r.class_id]
                # Prefix a stable track id (#N) when tracking is active.
                id_prefix = ""
                if self.enable_tracking and getattr(r, 'track_id', -1) >= 0:
                    id_prefix = f"#{r.track_id} "
                # For class-agnostic models (1 class), show instance index instead
                if len(self.labels) == 1:
                    text = f"{id_prefix}{label} {i+1} {r.confidence:.2f}"
                else:
                    text = f"{id_prefix}{label} {r.confidence:.2f}"

                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(output, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
                cv2.putText(output, text, (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Redraw masks for tracks that were missed this frame (no detection
        # above the score threshold) but are still within their hold window —
        # this is the actual flicker fix, not just decoration.
        if self.hold_frames > 0:
            expired = []
            for tid, held in self._held.items():
                if tid in seen_track_ids:
                    continue
                held["missed"] += 1
                if held["missed"] > self.hold_frames:
                    expired.append(tid)
                    continue
                self._draw_mask(overlay, held["mask"], held["box"],
                                 held["color"], img_w, img_h)
            for tid in expired:
                del self._held[tid]

        # Blend mask overlay (higher weight than a subtle tint so masks read
        # as solid pastel/neon fills, matching the reference look)
        output = cv2.addWeighted(overlay, 0.6, output, 0.4, 0)

        return output
