"""
Real-time brick segmentation using FastSAM.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import cv2
import torch
from ultralytics import FastSAM

from .base import BrickDetection, BrickSegmentorBase, visualize_detections, TARGET_CLASSES


class FastSAMSegmentor(BrickSegmentorBase):
    """
    Real-time brick segmentor backed by FastSAM. 
    Suitable for on-robot inference. 
    """

    CHECKPOINT_SMALL = "FastSAM-s.pt"
    CHECKPOINT_LARGE = "FastSAM-x.pt"

    CONFIDENCE_THRESHOLD = 0.80

    def __init__(self, large: bool = True, device: str = "cuda"):
        """
        Args:
            large:  If True, loads the larger FastSAM-x checkpoint.
            device: Torch device string ('cuda' or 'cpu').
        """
        self.device = device
        checkpoint = self.CHECKPOINT_LARGE if large else self.CHECKPOINT_SMALL

        print(f"Loading FastSAM ({checkpoint}) on {device}...")
        self.model = FastSAM(checkpoint)
        self.model.to(device)
        print("FastSAMSegmentor ready.")

    def segment(self, rgb_image: np.ndarray) -> list[BrickDetection]:
        """
        Detect and segment bricks in an RGB image.

        Args:
            rgb_image:      uint8 RGB image [H x W x 3].

        Returns:
            List of BrickDetection.
        """
        self._validate_image(rgb_image)

        results = self.model.predict(
            rgb_image,
            imgsz=640,
            conf=self.CONFIDENCE_THRESHOLD,
            device=self.device,
        )

        if not results or results[0].masks is None:
            return []

        result = results[0]
        masks = result.masks.data   # (N, H, W) tensor
        boxes = result.boxes

        detections: list[BrickDetection] = []
        for i in range(len(masks)):
            conf = float(boxes.conf[i].item())
            if conf < self.CONFIDENCE_THRESHOLD:
                continue

            class_name = self.model.names[int(boxes.cls[i].item())]
            # if class_name not in TARGET_CLASSES:
            #     continue

            mask = (masks[i] > 0.5).to(torch.uint8).cpu().numpy()
            bbox = boxes.xyxy[i].cpu().numpy()

            detections.append(BrickDetection(
                mask=mask,
                bbox=bbox,
                confidence=conf,
                class_name=class_name,
            ))

        return detections


# ===== Test =====

if __name__ == "__main__":
    import sys
    import time

    # Load test image
    if len(sys.argv) < 2:
        print("Usage: python segmentation.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    rgb_image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

    print("Running inference with `small` model.")
    segmentor = FastSAMSegmentor(large=False, device='cuda')

    # Warmup
    detections = segmentor.segment(rgb_image)
    
    # Test speed
    start = time.time()
    detections = segmentor.segment(rgb_image)
    duration = time.time() - start
    print(f"Speed: {len(detections)} detections in {duration * 1000:.1f}ms")
    
    # Visualize
    vis = visualize_detections(rgb_image, detections)
    cv2.imwrite(f'FastSAM_small_example_segmentation.jpg', cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    print("Running inference with `large` model.")
    segmentor = FastSAMSegmentor(large=True, device='cuda')

    # Warmup
    detections = segmentor.segment(rgb_image)
    
    # Test speed
    start = time.time()
    detections = segmentor.segment(rgb_image)
    duration = time.time() - start
    print(f"Speed: {len(detections)} detections in {duration * 1000:.1f}ms")
    
    # Visualize
    vis = visualize_detections(rgb_image, detections)
    cv2.imwrite(f'FastSAM_large_example_segmentation.jpg', cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    print("\nOutputs saved!")
