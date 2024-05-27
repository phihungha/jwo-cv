from __future__ import annotations

from dataclasses import dataclass

from cv2.typing import MatLike
from ultralytics import YOLO

from jwo_cv.utils import BoundingBox, Config

ITEM_DETECTOR_MODEL_PATH = "jwo_cv/config/item_yolov8n.pt"
HAND_DETECTOR_MODEL_PATH = "jwo_cv/config/hand_yolov8n.pt"
# TODO: Remove once the models are re-configured with no person class
PERSON_CLASS_ID = 0


@dataclass(frozen=True)
class Detection:
    """Describes an object detection with class ID, class name,
    confidence, and bounding box."""

    class_id: int
    class_name: str
    confidence: float
    box: BoundingBox

    def __str__(self) -> str:
        return (
            f"{{class_id: {self.class_id}, class_name: {self.class_name}, "
            f"confidence: {self.confidence}, box: {self.box}}}"
        )


class Detector:
    """Detects objects in images using Ultralytics YOLO."""

    def __init__(
        self,
        model: YOLO,
        min_confidence: float,
    ) -> None:
        """Detects objects in images using Ultralytics YOLO.

        Args:
            model (YOLO): Ultralytics YOLO model
            min_confidence (float): Minium detection confidence
        """

        self.model = model
        self.min_confidence = min_confidence

    def detect(self, image: MatLike) -> list[Detection]:
        """Detect and classify objects in an image.

        Args:
            image (MatLike): Image

        Returns:
            list[ItemDetection]: Detections
        """

        outputs = self.model.predict(image, verbose=False)
        detections: list[Detection] = []

        for output in outputs:
            if output.boxes is None:
                continue

            for result in output.boxes:
                confidence = float(result.conf)
                if confidence < self.min_confidence:
                    continue

                class_id = int(result.cls)
                class_name = self.model.names[class_id]
                box = BoundingBox.from_xyxy_arr(result.xyxy[0])
                detection = Detection(class_id, class_name, confidence, box)
                detections.append(detection)

        return detections


class HandDetector(Detector):
    """Detects hands in an image."""

    @classmethod
    def from_config(cls, config: Config) -> HandDetector:
        model = YOLO(HAND_DETECTOR_MODEL_PATH)
        return cls(
            model,
            config["min_confidence"],
        )


class ItemDetector(Detector):
    """Detects and classifies product items in hands in an image."""

    def __init__(
        self,
        model: YOLO,
        hand_detector: HandDetector,
        min_confidence: float,
        max_hand_distance: float,
    ) -> None:
        """Detects and classifies product items hold in hands in an image.

        Args:
            model (YOLO): Ultralytics YOLO model
            hand_detector (HandDetector): Hand detector
            min_confidence (float): Minium detection confidence
            max_hand_distance (float): Max distance from hands to detect
        """

        super().__init__(model, min_confidence)
        self.hand_detector = hand_detector
        self.max_hand_distance = max_hand_distance

    @classmethod
    def from_config(cls, config: Config) -> ItemDetector:
        model = YOLO(ITEM_DETECTOR_MODEL_PATH)
        hand_detector = HandDetector.from_config(config["hand"])
        return cls(
            model,
            hand_detector,
            config["item"]["min_confidence"],
            config["item"]["max_hand_distance"],
        )

    def detect(self, image: MatLike) -> tuple[list[Detection], list[Detection]]:
        """Detects and classifies items hold in hands in an image.

        Args:
            image (MatLike): Image

        Returns:
            tuple[list[Detection], list[Detection]]: Item and hand detections
        """

        items = super().detect(image)
        hands = self.hand_detector.detect(image)

        def filter_item(item: Detection):
            # TODO: Remove once the models are re-configured with no person class
            is_not_person = item.class_id != PERSON_CLASS_ID
            is_near_hand = any(
                item.box.calc_distance(hand.box) <= self.max_hand_distance
                for hand in hands
            )
            return is_not_person and is_near_hand

        return list(filter(filter_item, items)), hands
