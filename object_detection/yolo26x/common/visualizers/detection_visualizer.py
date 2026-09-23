"""
Detection Visualizer

Draws bounding boxes and labels for object detection results.
Used by YOLO family, SCRFD, etc.
"""

import numpy as np
import cv2
from typing import List

from ..base import IVisualizer, DetectionResult
from ..utility import get_labels


class DetectionVisualizer(IVisualizer):
    """
    Visualizer for object detection results.
    
    Draws bounding boxes with class labels and confidence scores.
    """

    # Horizontal gap (px) between the label background edge and the text
    # itself, on both sides. Doubled from the ~3px the old flush-left layout
    # effectively had.
    _LABEL_PAD_X = 6

    def __init__(self, label_set: str = 'coco80', custom_labels: List[str] = None):
        """
        Initialize detection visualizer.
        
        Args:
            label_set: Predefined label set name ('coco80', 'coco', 'voc', etc.)
            custom_labels: Custom label list (overrides label_set)
        """
        if custom_labels is not None:
            self.labels = custom_labels
        else:
            self.labels = get_labels(label_set)

        # Pastel color per class: hues spread evenly around the wheel, with
        # low saturation and high value so every class reads as soft/light
        # rather than the harsh fully-saturated colors np.random.uniform gave.
        self.color_palette = self._generate_pastel_palette(len(self.labels))

    @staticmethod
    def _generate_pastel_palette(count: int) -> np.ndarray:
        """Return `count` pastel BGR colors, evenly spaced by hue."""
        hues = (np.arange(count) * 179 / max(count, 1)).astype(np.uint8)
        hsv = np.zeros((count, 1, 3), dtype=np.uint8)
        hsv[:, 0, 0] = hues
        hsv[:, 0, 1] = 90   # low-medium saturation -> pastel
        hsv[:, 0, 2] = 255  # high value -> bright
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return bgr[:, 0, :].astype(np.float64)
    
    def visualize(self, image: np.ndarray, results: List[DetectionResult]) -> np.ndarray:
        """
        Draw detection results on image.

        Labels are anchored to whichever top corner of the box sits further
        from the frame's vertical centerline: boxes on the left half get
        their label at the box's top-left, boxes on the right half get it at
        the box's top-right. This keeps labels from piling up in the middle
        of the frame where left- and right-side detections would otherwise
        overlap.

        Args:
            image: Original image (BGR format)
            results: List of DetectionResult objects

        Returns:
            Image with drawn detections
        """
        output = image.copy()
        frame_center_x = image.shape[1] / 2

        for det in results:
            x1, y1, x2, y2 = [int(v) for v in det.box]
            class_id = det.class_id

            # Get color
            color = self.color_palette[class_id % len(self.color_palette)]

            # Draw bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            # Prepare label text (class name only, no confidence score)
            label = self.labels[class_id] if class_id < len(self.labels) else f"class_{class_id}"

            # Get text size
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            # Anchor to top-left for boxes on the left half, top-right for
            # boxes on the right half (box center vs. frame centerline).
            box_center_x = (x1 + x2) / 2
            anchor_right = box_center_x >= frame_center_x
            label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

            # Horizontal padding between the background edge and the text
            # itself (symmetric on both sides of the label).
            pad_x = self._LABEL_PAD_X
            if anchor_right:
                bg_x2 = x2
                bg_x1 = bg_x2 - label_width - 2 * pad_x
            else:
                bg_x1 = x1
                bg_x2 = bg_x1 + label_width + 2 * pad_x
            text_x = bg_x1 + pad_x

            # Draw label background (bottom follows the text baseline, not a
            # full second line of height, so there's no dead space below the
            # glyphs)
            cv2.rectangle(
                output,
                (bg_x1, label_y - label_height),
                (bg_x2, label_y + baseline),
                color,
                cv2.FILLED
            )

            # Draw label text
            cv2.putText(
                output,
                label,
                (text_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )

        return output
