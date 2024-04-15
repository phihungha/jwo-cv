from dataclasses import dataclass
from typing import Sequence
from cv2.typing import MatLike
import torch
from ultralytics import YOLO
from jwo_cv import utils

PERSON_CLASS_NAME = "person"


@dataclass(frozen=True)
class HandDetection:
    """Describes a hand detection with confidence and bounding box."""

    confidence: float
    box: utils.BoundingBox


@dataclass(frozen=True)
class ItemDetection:
    """Describes an item detection with class name,
    confidence, and bounding box."""

    class_name: str
    confidence: float
    box: utils.BoundingBox


class HandDetector:
    """Detects hands in an image."""

    def predict(self) -> list[HandDetection]:
        """Detect hands in an image.

        Returns:
            list[HandDetection]: Detections
        """

        hands: list[HandDetection] = []
        confidence = float(torch.Tensor([1]))
        box = utils.BoundingBox.from_xyxy_tensor(torch.Tensor([250, 250, 400, 400]))
        pred_result = HandDetection(confidence, box)
        hands.append(pred_result)

        return hands


class ItemDetector:
    """Detects and classifies product items in an image."""

    def __init__(self, model_path: str, max_hand_distance: int | float) -> None:
        """Detect and classifies product items in an image.

        Args:
            model_path (str): Path to model config file
            max_hand_distance (int | float): Max distance from hands to detect items
        """

        self.model = YOLO(model_path)
        self.max_hand_distance = max_hand_distance

    def detect(
        self, image: MatLike, mask_boxes: Sequence[utils.BoundingBox]
    ) -> list[ItemDetection]:
        """Detect and classify items in a box region of an image.

        Args:
            image (MatLike): Image
            mask_boxes (Sequence[utils.BoundingBox]): Region box to look in

        Returns:
            list[ItemDetection]: Detections
        """

        results = self.model.predict(image, verbose=False)

        detections: list[ItemDetection] = []
        for result in results:
            for result_box in result.boxes:
                confidence = float(result_box.conf)

                class_name = self.model.names[int(result_box.cls)]
                # TODO: Re-configure model without person class
                if class_name == PERSON_CLASS_NAME:
                    continue

                box = utils.BoundingBox.from_xyxy_tensor(result_box.xyxy[0])
                isNearHand = any(
                    box.calcDistance(mask) < self.max_hand_distance
                    for mask in mask_boxes
                )
                if not isNearHand:
                    continue

                detection = ItemDetection(class_name, confidence, box)
                detections.append(detection)

        return detections
