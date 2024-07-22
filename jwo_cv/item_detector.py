from __future__ import annotations

from dataclasses import dataclass

from cv2.typing import MatLike
from ultralytics import YOLO

from jwo_cv.utils import BoundingBox, Config


@dataclass(frozen=True)
class Detection:
    """Describes an object detection with class ID, class name,
    confidence, and bounding box."""

    class_id: int
    class_name: str
    confidence: float
    box: BoundingBox
    box_normed: BoundingBox

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
                print(result.xyxyn[0])
                box = BoundingBox.from_xyxy_arr(result.xyxy[0])
                box_normed = BoundingBox.from_xyxy_arr(result.xyxyn[0])
                detection = Detection(class_id, class_name, confidence, box, box_normed)
                detections.append(detection)

        return detections


class HandDetector(Detector):
    """Detects hands in an image."""

    @classmethod
    def from_config(cls, config: Config) -> HandDetector:
        return cls(
            YOLO(config["model_path"]),
            config["min_confidence"],
        )


class ItemDetector(Detector):
    """Detects and classifies product items in hands in an image."""

    def __init__(
        self,
        model: YOLO,
        hand_detector: HandDetector,
        min_confidence: float,
        min_hand_iou: float,
    ) -> None:
        """Detects and classifies product items hold in hands in an image.

        Args:
            model (YOLO): Ultralytics YOLO model
            hand_detector (HandDetector): Hand detector
            min_confidence (float): Minium detection confidence
            min_hand_iou (float): Min IoU of item and hand bounding boxes
        """

        super().__init__(model, min_confidence)
        self.hand_detector = hand_detector
        self.min_hand_iou = min_hand_iou

    @classmethod
    def from_config(cls, config: Config) -> ItemDetector:
        hand_detector = HandDetector.from_config(config["hand"])
        return cls(
            YOLO(config["item"]["model_path"]),
            hand_detector,
            config["item"]["min_confidence"],
            config["item"]["min_hand_iou"],
        )

    def detect(self, image: MatLike) -> list[Detection]:
        """Detects and classifies items hold in hands in an image.

        Args:
            image (MatLike): Image

        Returns:
            list[Detection]: Item detections
        """

        hands = self.hand_detector.detect(image)
        if not hands:
            return []

        items = super().detect(image)

        def filter_item(item: Detection):
            return any(
                item.box.calc_iou(hand.box) >= self.min_hand_iou for hand in hands
            )

        items_in_hands = list(filter(filter_item, items))

        return items_in_hands
