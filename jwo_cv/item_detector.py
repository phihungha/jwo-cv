from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from cv2.typing import MatLike
from ultralytics import YOLO
from jwo_cv import utils

PERSON_LABEL = "person"


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
        model: YOLO,
        min_confidence: float,
    ) -> None:
        """An OpenCV-based object detector.

        Args:
            model (YOLO): Ultralytics YOLO model
            image_size (utils.ImageSize): Size of image to detect from
            min_confidence (float): Minimum confidence of detections to use
        """

        self.model = model
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


class HandDetector(Detector):
    """Detects hands in an image."""

    @classmethod
    def from_config(cls, config: Mapping) -> HandDetector:
        model = YOLO(config["model_path"])
        return cls(
            model,
            config["min_confidence"],
        )

    def detect(self, image: MatLike) -> list[Detection]:
        """Detect hands.

        Args:
            image (MatLike): Image

        Returns:
            list[ItemDetection]: Detections
        """

        detections = super().detect(image)
        return list(filter(lambda d: d.class_name == PERSON_LABEL, detections))


class ItemDetector(Detector):
    """Detects and classifies product items in an image."""

    def __init__(
        self,
        model: YOLO,
        min_confidence: float,
        max_hand_distance: float,
    ) -> None:
        """Detect and classifies product items in an image.

        Args:
            model (str): Ultralytics YOLO model
            min_confidence (float): Minimum confidence of detections to use
            max_hand_distance (float): Max distance from hands to detect items
        """

        super().__init__(model, min_confidence)
        self.max_hand_distance = max_hand_distance

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> ItemDetector:
        model = YOLO(config["model_path"])
        return cls(
            model,
            config["min_confidence"],
            config["max_hand_distance"],
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

        def filter_item(detection: Detection):
            # Re-configure model without person class
            is_not_person = detection.class_name != PERSON_LABEL
            is_near_hand = any(
                detection.box.calc_distance(box) <= self.max_hand_distance
                for box in mask_boxes
            )
            return is_not_person and is_near_hand

        return list(filter(filter_item, detections))
