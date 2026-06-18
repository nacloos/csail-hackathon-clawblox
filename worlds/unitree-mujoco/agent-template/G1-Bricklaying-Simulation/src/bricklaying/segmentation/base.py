"""
Shared types and base class for brick segmentation.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import List

import numpy as np
import cv2


TARGET_CLASSES = ["brick", "block"]


@dataclass
class BrickDetection:
    """Single brick detection result."""
    mask: np.ndarray      # Binary mask [H x W], uint8
    bbox: np.ndarray      # Bounding box [x_min, y_min, x_max, y_max]
    confidence: float     # Detection confidence [0, 1]
    class_name: str       # Detected class label (e.g. "brick", "block")


class BrickSegmentorBase(abc.ABC):
    """
    Common interface for brick segmentation models.

    Subclasses must implement `segment`, which accepts a uint8 RGB numpy
    array and returns a list of BrickDetection objects — regardless of
    whatever internal format the underlying model prefers.
    """

    @abc.abstractmethod
    def segment(self, rgb_image: np.ndarray) -> list[BrickDetection]:
        """
        Detect and segment bricks in an RGB image.

        Args:
            rgb_image: uint8 RGB image [H x W x 3].

        Returns:
            List of BrickDetection, one per detected brick.
        """

    # ------------------------------------------------------------------
    # Shared input validation — call from subclass segment() methods
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_image(rgb_image: np.ndarray) -> None:
        if rgb_image.dtype != np.uint8:
            raise ValueError("rgb_image must be uint8")
        if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
            raise ValueError("rgb_image must have shape [H, W, 3]")
        

def visualize_detections(rgb_image: np.ndarray,
                        detections: List[BrickDetection],
                        alpha: float = 0.5) -> np.ndarray:
    """Overlay detection masks and labels on image."""
    vis = rgb_image.copy()
    
    colors = [
        [255, 0, 0], [0, 255, 0], [0, 0, 255],
        [255, 255, 0], [255, 0, 255], [0, 255, 255]
    ]
    
    for i, det in enumerate(detections):
        color = colors[i % len(colors)]
        
        # Draw mask
        mask_colored = np.zeros_like(rgb_image)
        mask_colored[det.mask > 0.5] = color
        vis = cv2.addWeighted(vis, 1.0, mask_colored, alpha, 0)
        
        # Draw bbox
        x_min, y_min, x_max, y_max = det.bbox.astype(int)
        cv2.rectangle(vis, (x_min, y_min), (x_max, y_max), color, 2)
        
        # Draw label
        label = f"{det.class_name}: {det.confidence:.2f}"
        cv2.putText(vis, label, (x_min, y_min - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return vis