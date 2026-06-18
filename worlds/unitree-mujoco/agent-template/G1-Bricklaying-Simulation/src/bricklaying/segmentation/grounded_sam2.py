"""
High-quality "ground truth" brick segmentation using Grounded-SAM2.

Intended for offline analysis and evaluation — not real-time use.
Requires installing dependencies manually, e.g.:
    git clone https://github.com/IDEA-Research/Grounded-SAM-2.git && \
    pip install -e Grounded-SAM-2 && \
    pip install --no-build-isolation -e Grounded-SAM-2/grounding_dino && \
    cd Grounded-SAM-2/checkpoints && bash download_ckpts.sh && \
    cd ../gdino_checkpoints && bash download_ckpts.sh

To-do:
    - SAM3 supports text-based prompting natively; evaluate as a replacement.
"""

from __future__ import annotations

import numpy as np
import cv2
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from .base import BrickDetection, BrickSegmentorBase, visualize_detections, TARGET_CLASSES


torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


class GroundedSAM2Segmentor(BrickSegmentorBase):
    """
    High-quality brick segmentor backed by Grounding-DINO + SAM2.

    Heavier than FastSAM but produces more accurate masks, making it
    suitable as a reference for offline evaluation.
    """

    SAM2_CHECKPOINT = "../Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
    SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
    GROUNDING_MODEL = "IDEA-Research/grounding-dino-base"

    CONFIDENCE_THRESHOLD = 0.4

    def __init__(self, device: str = "cuda"):
        self.device = device

        print("Loading Grounded-SAM2...")
        sam2_model = build_sam2(self.SAM2_MODEL_CONFIG, self.SAM2_CHECKPOINT, device=device)
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)
        self.processor = AutoProcessor.from_pretrained(self.GROUNDING_MODEL)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.GROUNDING_MODEL
        ).to(device)
        print("GroundedSAM2Segmentor ready.")

    def segment(self, rgb_image: np.ndarray) -> list[BrickDetection]:
        """
        Detect and segment bricks in an RGB image.

        Args:
            rgb_image: uint8 RGB image [H x W x 3].

        Returns:
            List of BrickDetection.
        """
        self._validate_image(rgb_image)

        image_pil = Image.fromarray(rgb_image)
        self.sam2_predictor.set_image(rgb_image)

        inputs = self.processor(
            images=image_pil,
            text=TARGET_CLASSES,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.grounding_model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.CONFIDENCE_THRESHOLD,
            text_threshold=self.CONFIDENCE_THRESHOLD,
            target_sizes=[image_pil.size[::-1]],
        )

        boxes = results[0]["boxes"]
        confidences = results[0]["scores"].cpu().numpy().tolist()
        labels = results[0]["labels"]

        if len(boxes) == 0:
            return []

        masks, _, _ = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes,
            multimask_output=False,
        )

        if masks.ndim == 4:
            masks = masks.squeeze(1)

        masks = (masks > 0.5).astype(np.uint8)

        return [
            BrickDetection(
                mask=masks[i],
                bbox=boxes[i].cpu().numpy(),
                confidence=confidences[i],
                class_name=labels[i],
            )
            for i in range(len(masks))
        ]


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

    print("Running inference with Grounded-SAM2.")
    segmentor = GroundedSAM2Segmentor(device='cuda')

    # Warmup
    detections = segmentor.segment(rgb_image)
    
    # Test speed
    start = time.time()
    detections = segmentor.segment(rgb_image)
    duration = time.time() - start
    print(f"Speed: {len(detections)} detections in {duration * 1000:.1f}ms")
    
    # Visualize
    vis = visualize_detections(rgb_image, detections)
    cv2.imwrite(f'Grounded_SAM2_example_segmentation.jpg', cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    print("\nOutputs saved!")
