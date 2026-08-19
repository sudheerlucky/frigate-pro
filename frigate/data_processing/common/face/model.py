import logging
import os
import queue
import threading
from abc import ABC, abstractmethod

import cv2
import numpy as np

from frigate.config import FrigateConfig
from frigate.const import FACE_DIR, MODEL_CACHE_DIR
from frigate.embeddings.onnx.face_embedding import ArcfaceEmbedding, FaceNetEmbedding
from frigate.log import redirect_output_to_logger

logger = logging.getLogger(__name__)


class FaceRecognizer(ABC):
    """Face recognition runner."""

    def __init__(self, config: FrigateConfig) -> None:
        self.config = config
        self.landmark_detector: cv2.face.Facemark | None = None
        self.init_landmark_detector()

    @abstractmethod
    def build(self) -> None:
        """Build face recognition model."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear current built model."""
        pass

    @abstractmethod
    def classify(self, face_image: np.ndarray) -> tuple[str, float] | None:
        pass

    @redirect_output_to_logger(logger, logging.DEBUG)  # type: ignore[misc]
    def init_landmark_detector(self) -> None:
        landmark_model = os.path.join(MODEL_CACHE_DIR, "facedet/landmarkdet.yaml")

        if os.path.exists(landmark_model):
            landmark_detector = cv2.face.createFacemarkLBF()
            landmark_detector.loadModel(landmark_model)
            self.landmark_detector = landmark_detector

    def estimate_face_yaw_ratio(self, image: np.ndarray) -> float | None:
        """Rough frontality estimate from the same 68-point landmark model
        align_face uses, without doing the full alignment: ratio of the
        nose tip's horizontal distance to each eye's center. ~1.0 for a
        face looking straight at the camera; a strongly asymmetric ratio
        means one eye is much closer to the nose than the other in the
        2D projection -- a full side-profile view, which face embedding
        models trained on frontal-ish faces don't reliably identify
        people from even at a high raw similarity score. Returns None if
        landmarks can't be computed at all (e.g. no landmark detector
        loaded, or the crop is unusable for a different reason)."""
        if not self.landmark_detector:
            return None

        land_image = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        )

        try:
            ok, lands = self.landmark_detector.fit(
                land_image,
                np.array([(0, 0, land_image.shape[1], land_image.shape[0])]),
            )
        except cv2.error:
            return None

        if not ok:
            return None

        landmarks: np.ndarray = lands[0][0]
        left_eye_center = landmarks[42:48].mean(axis=0)
        right_eye_center = landmarks[36:42].mean(axis=0)
        nose_tip = landmarks[30]

        dist_to_left = abs(float(nose_tip[0] - left_eye_center[0]))
        dist_to_right = abs(float(nose_tip[0] - right_eye_center[0]))

        if dist_to_left < 1.0 or dist_to_right < 1.0:
            return 999.0  # degenerate geometry, treat as non-frontal

        return max(dist_to_left, dist_to_right) / min(dist_to_left, dist_to_right)

    def align_face(
        self,
        image: np.ndarray,
        output_width: int,
        output_height: int,
    ) -> np.ndarray:
        if not self.landmark_detector:
            raise ValueError("Landmark detector not initialized")

        # landmark is run on grayscale images
        if image.ndim == 3:
            land_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            land_image = image

        _, lands = self.landmark_detector.fit(
            land_image, np.array([(0, 0, land_image.shape[1], land_image.shape[0])])
        )
        landmarks: np.ndarray = lands[0][0]

        # get landmarks for eyes
        leftEyePts = landmarks[42:48]
        rightEyePts = landmarks[36:42]

        # compute the center of mass for each eye
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - 0.35

        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX**2) + (dY**2))
        desiredDist = desiredRightEyeX - 0.35
        desiredDist *= output_width
        scale = desiredDist / dist

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        # grab the rotation matrix for rotating and scaling the face
        eyesCenter = (
            int((leftEyeCenter[0] + rightEyeCenter[0]) // 2),
            int((leftEyeCenter[1] + rightEyeCenter[1]) // 2),
        )
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

        # update the translation component of the matrix
        tX = output_width * 0.5
        tY = output_height * 0.35
        M[0, 2] += tX - eyesCenter[0]
        M[1, 2] += tY - eyesCenter[1]

        # apply the affine transformation
        return cv2.warpAffine(
            image, M, (output_width, output_height), flags=cv2.INTER_CUBIC
        )

    def get_blur_confidence_reduction(self, input: np.ndarray) -> float:
        """Calculates the reduction in confidence based on the blur of the image."""
        if not self.config.face_recognition.blur_confidence_filter:
            return 0.0

        variance = cv2.Laplacian(input, cv2.CV_64F).var()
        logger.debug(f"face detected with blurriness {variance}")

        if variance < 120:  # image is very blurry
            return 0.06
        elif variance < 160:  # image moderately blurry
            return 0.04
        elif variance < 200:  # image is slightly blurry
            return 0.02
        elif variance < 250:  # image is mostly clear
            return 0.01
        else:
            return 0.0


def build_class_references(
    embs: list[np.ndarray],
    outlier_threshold: float = 0.30,
    min_keep_frac: float = 0.7,
    max_iters: int = 3,
) -> list[np.ndarray]:
    """Prune outlier embeddings but keep each surviving reference as its own
    vector instead of collapsing them into a single class-mean.

    A single averaged prototype blends dissimilar reference photos (a sharp
    close-up next to a dim, blurry camera crop) into a vector that resembles
    neither well. Matching a live face against the closest individual
    reference (nearest-neighbor, like this) instead of one blended average
    lets a single clean, representative photo win a match on its own,
    rather than being diluted by lower-quality ones in the same folder.

    Uses the same iterative vector-wise outlier rejection as
    ``build_class_mean`` to still drop mislabeled/corrupted samples.
    """
    arr = np.stack(embs, axis=0)

    if len(arr) < 5:
        return list(arr)

    keep = np.ones(len(arr), dtype=bool)
    floor = max(5, int(np.ceil(min_keep_frac * len(arr))))

    for _ in range(max_iters):
        mean = arr[keep].mean(axis=0)
        m_norm = mean / (np.linalg.norm(mean) + 1e-9)
        e_norms = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)
        cos = e_norms @ m_norm
        new_keep = cos >= outlier_threshold

        if new_keep.sum() < floor:
            top = np.argsort(-cos)[:floor]
            new_keep = np.zeros(len(arr), dtype=bool)
            new_keep[top] = True

        if np.array_equal(new_keep, keep):
            break
        keep = new_keep

    dropped = int((~keep).sum())

    if dropped:
        logger.debug(
            f"Vector-wise outlier filter dropped {dropped}/{len(arr)} embeddings"
        )

    return list(arr[keep])


def best_match_confidence(
    embedding: np.ndarray, references: list[np.ndarray], top_k: int = 3
) -> float:
    """Cosine similarity of a probe embedding against the top-k closest
    individual reference embeddings for a person, averaged.

    Averaging a small top-k (rather than taking the single best match)
    guards against one lucky near-duplicate reference frame producing a
    falsely high score, while still letting one clean reference dominate a
    match instead of being dragged down by an unrelated blended average.
    """
    magnitude_probe = np.linalg.norm(embedding)
    sims = [
        float(np.dot(embedding, ref) / (magnitude_probe * np.linalg.norm(ref) + 1e-9))
        for ref in references
    ]
    k = min(top_k, len(sims))
    top = sorted(sims, reverse=True)[:k]
    return sum(top) / k


def similarity_to_confidence(
    cosine_similarity: float,
    median: float = 0.3,
    range_width: float = 0.6,
    slope_factor: float = 12,
) -> float:
    """
    Default sigmoid function to map cosine similarity to confidence.

    Args:
        cosine_similarity (float): The input cosine similarity.
        median (float): Assumed median of cosine similarity distribution.
        range_width (float): Assumed range of cosine similarity distribution (90th percentile - 10th percentile).
        slope_factor (float): Adjusts the steepness of the curve.

    Returns:
        float: The confidence score.
    """

    # Calculate slope and bias
    slope = slope_factor / range_width
    bias = median

    # Calculate confidence
    confidence: float = 1 / (1 + np.exp(-slope * (cosine_similarity - bias)))
    return confidence


class FaceNetRecognizer(FaceRecognizer):
    def __init__(self, config: FrigateConfig):
        super().__init__(config)
        self.class_refs: dict[str, list[np.ndarray]] = {}
        self.face_embedder: FaceNetEmbedding = FaceNetEmbedding()
        self.model_builder_queue: queue.Queue | None = None

    def clear(self) -> None:
        self.class_refs = {}

    def run_build_task(self) -> None:
        self.model_builder_queue = queue.Queue()

        def build_model() -> None:
            face_embeddings_map: dict[str, list[np.ndarray]] = {}
            idx = 0

            dir = FACE_DIR
            for name in os.listdir(dir):
                if name == "train":
                    continue

                face_folder = os.path.join(dir, name)

                if not os.path.isdir(face_folder):
                    continue

                face_embeddings_map[name] = []
                for image in os.listdir(face_folder):
                    img = cv2.imread(os.path.join(face_folder, image))

                    if img is None:
                        continue  # type: ignore[unreachable]

                    img = self.align_face(img, img.shape[1], img.shape[0])
                    emb = self.face_embedder([img])[0].squeeze()
                    face_embeddings_map[name].append(emb)

                idx += 1

            assert self.model_builder_queue is not None
            self.model_builder_queue.put(face_embeddings_map)

        thread = threading.Thread(target=build_model, daemon=True)
        thread.start()

    def build(self) -> None:
        if not self.landmark_detector:
            self.init_landmark_detector()
            return None

        if self.model_builder_queue is not None:
            try:
                face_embeddings_map: dict[str, list[np.ndarray]] = (
                    self.model_builder_queue.get(timeout=0.1)
                )
                self.model_builder_queue = None
            except queue.Empty:
                return
        else:
            self.run_build_task()
            return

        if not face_embeddings_map:
            return

        for name, embs in face_embeddings_map.items():
            if embs:
                self.class_refs[name] = build_class_references(embs)

        logger.debug("Finished building ArcFace model")

    def classify(self, face_image: np.ndarray) -> tuple[str, float] | None:
        if not self.landmark_detector:
            return None

        if not self.class_refs:
            self.build()

            if not self.class_refs:
                return None

        # face recognition is best run on grayscale images

        # get blur factor before aligning face
        blur_reduction = self.get_blur_confidence_reduction(face_image)

        # align face and run recognition
        img = self.align_face(face_image, face_image.shape[1], face_image.shape[0])
        embedding = self.face_embedder([img])[0].squeeze()

        score: float = 0
        label = ""

        for name, refs in self.class_refs.items():
            cosine_similarity = best_match_confidence(embedding, refs)
            confidence = similarity_to_confidence(
                cosine_similarity, median=0.5, range_width=0.6
            )

            if confidence > score:
                score = confidence
                label = name

        return label, max(0, round(score - blur_reduction, 2))


class ArcFaceRecognizer(FaceRecognizer):
    def __init__(self, config: FrigateConfig):
        super().__init__(config)
        self.class_refs: dict[str, list[np.ndarray]] = {}
        self.face_embedder: ArcfaceEmbedding = ArcfaceEmbedding(config.face_recognition)
        self.model_builder_queue: queue.Queue | None = None

    def clear(self) -> None:
        self.class_refs = {}

    def run_build_task(self) -> None:
        self.model_builder_queue = queue.Queue()

        def build_model() -> None:
            face_embeddings_map: dict[str, list[np.ndarray]] = {}
            idx = 0

            dir = FACE_DIR
            for name in os.listdir(dir):
                if name == "train":
                    continue

                face_folder = os.path.join(dir, name)

                if not os.path.isdir(face_folder):
                    continue

                face_embeddings_map[name] = []
                for image in os.listdir(face_folder):
                    img = cv2.imread(os.path.join(face_folder, image))

                    if img is None:
                        continue  # type: ignore[unreachable]

                    img = self.align_face(img, img.shape[1], img.shape[0])
                    emb = self.face_embedder([img])[0].squeeze()  # type: ignore[arg-type]
                    face_embeddings_map[name].append(emb)

                idx += 1

            assert self.model_builder_queue is not None
            self.model_builder_queue.put(face_embeddings_map)

        thread = threading.Thread(target=build_model, daemon=True)
        thread.start()

    def build(self) -> None:
        if not self.landmark_detector:
            self.init_landmark_detector()
            return None

        if self.model_builder_queue is not None:
            try:
                face_embeddings_map: dict[str, list[np.ndarray]] = (
                    self.model_builder_queue.get(timeout=0.1)
                )
                self.model_builder_queue = None
            except queue.Empty:
                return
        else:
            self.run_build_task()
            return

        if not face_embeddings_map:
            return

        for name, embs in face_embeddings_map.items():
            if embs:
                self.class_refs[name] = build_class_references(embs)

        logger.debug("Finished building ArcFace model")

    def classify(self, face_image: np.ndarray) -> tuple[str, float] | None:
        if not self.landmark_detector:
            return None

        if not self.class_refs:
            self.build()

            if not self.class_refs:
                return None

        # face recognition is best run on grayscale images

        # get blur reduction before aligning face
        blur_reduction = self.get_blur_confidence_reduction(face_image)

        # align face and run recognition
        img = self.align_face(face_image, face_image.shape[1], face_image.shape[0])
        embedding = self.face_embedder([img])[0].squeeze()  # type: ignore[arg-type]

        score: float = 0
        label = ""

        for name, refs in self.class_refs.items():
            cosine_similarity = best_match_confidence(embedding, refs)
            confidence = similarity_to_confidence(cosine_similarity)

            if confidence > score:
                score = confidence
                label = name

        return label, max(0, round(score - blur_reduction, 2))
