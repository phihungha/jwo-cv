from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from cv2.typing import MatLike
import numpy as np
from ultralytics import YOLO
from jwo_cv import utils


@dataclass(frozen=True)
class Detection:
    """Describes an object detection with class name,
    confidence, and bounding box."""

    class_name: str
    confidence: float
    box: utils.BoundingBox


class Detector:
    """An OpenCV-based object detector."""

    def __init__(
        self,
        model_path: str,
        min_confidence: float,
    ) -> None:
        """An OpenCV-based object detector.

        Args:
            model_path (str): Path to model file
            image_size (utils.ImageSize): Size of image to detect from
            min_confidence (float): Minimum confidence of detections to use
        """

        self.model = YOLO(model_path)
        self.min_confidence = min_confidence

    def detect(self, image: MatLike) -> list[Detection]:
        """Detect and classify items on an image.

        Args:
            image (MatLike): Image

        Returns:
            list[ItemDetection]: Detections
        """

        outputs = self.model.predict(image, verbose=False)
        detections: list[Detection] = []

        for output in outputs:
            for result in output.boxes:
                confidence = float(result.conf)
                if confidence < self.min_confidence:
                    continue

                class_name = self.model.names[int(result.cls)]
                box = utils.BoundingBox.from_xyxy_arr(result.xyxy[0])
                detection = Detection(class_name, confidence, box)
                detections.append(detection)

        return detections


class HandDetector:
    """Detects hands in an image."""

    def detect(self) -> list[Detection]:
        """Detect hands in an image.

        Returns:
            list[Detection]: Detections
        """

        hands: list[Detection] = []
        confidence = 1
        box = utils.BoundingBox.from_xyxy_arr(np.array([300, 300, 150, 150]))
        pred_result = Detection("hand", confidence, box)
        hands.append(pred_result)

        return hands


class ItemDetector(Detector):
    """Detects and classifies product items in an image."""

    def __init__(
        self,
        model_path: str,
        min_confidence: float,
        max_hand_distance: float,
        exclude_class_names: Sequence[str],
    ) -> None:
        """Detect and classifies product items in an image.

        Args:
            model_path (str): Path to model file
            image_size (utils.ImageSize): Size of image to detect from
            min_confidence (float): Minimum confidence of detections to use
            max_hand_distance (float): Max distance from hands to detect items
        """

        super().__init__(model_path, min_confidence)
        self.max_hand_distance = max_hand_distance
        self.exclude_class_name = exclude_class_names

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> ItemDetector:
        return cls(
            config["model_path"],
            config["min_confidence"],
            config["max_hand_distance"],
            config["exclude_class_names"],
        )

    def detect(
        self, image: MatLike, mask_boxes: Sequence[utils.BoundingBox]
    ) -> list[Detection]:
        """Detect and classify items in provided region boxes of an image.

        Args:
            image (MatLike): Image
            mask_boxes (Sequence[utils.BoundingBox]): Region boxes to look in

        Returns:
            list[ItemDetection]: Detections
        """

        detections = super().detect(image)

        def filterItem(detection: Detection):
            if detection.class_name in self.exclude_class_name:
                return False

            if not any(
                detection.box.calcDistance(mask) <= self.max_hand_distance
                for mask in mask_boxes
            ):
                return False

            return True

        return list(filter(filterItem, detections))
