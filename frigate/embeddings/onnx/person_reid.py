"""Dedicated person re-identification embeddings via OSNet (x0_25, MSMT17
fine-tuned), converted from the original MIT-licensed PyTorch weights
(github.com/KaiyangZhou/deep-person-reid) to ONNX.

This is a real re-ID model -- trained specifically to distinguish people by
appearance across camera views -- as opposed to the general-purpose CLIP
embeddings semantic search already computes. More precise for cross-camera
person matching, but still fundamentally a similarity heuristic: results
are reported as confidence scores, never as confirmed identity.
"""

import logging
import os

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "models", "osnet_x0_25_msmt17.onnx"
)

INPUT_HEIGHT = 256
INPUT_WIDTH = 128
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class PersonReidEmbedding:
    """Loads once, embeds many crops. Not thread-safe for concurrent
    `embed()` calls on the same instance -- create one per caller if run
    concurrently, or serialize calls."""

    def __init__(self) -> None:
        self.session = ort.InferenceSession(
            MODEL_PATH, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def _preprocess(self, image_bytes: bytes) -> np.ndarray | None:
        img = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT))
        img = img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        return np.expand_dims(img, axis=0).astype(np.float32)

    def embed(self, image_bytes: bytes) -> np.ndarray | None:
        """Returns a 512-d embedding for a person crop, or None if the
        image couldn't be decoded."""
        tensor = self._preprocess(image_bytes)
        if tensor is None:
            return None

        result = self.session.run(None, {self.input_name: tensor})
        return result[0][0]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
