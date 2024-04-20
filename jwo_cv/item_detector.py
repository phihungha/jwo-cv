from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import cv2
from cv2.typing import MatLike
import numpy as np
import yaml
from jwo_cv import utils

PERSON_CLASS_ID = 0


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
        labels: Mapping[int, str],
        image_size: utils.Size,
        min_confidence: float,
    ) -> None:
        """An OpenCV-based object detector.

        Args:
            model_path (str): Path to model file
            labels (Mapping[int, str]): Class-to-label map
            image_size (utils.ImageSize): Size of image to detect from
            min_confidence (float): Minimum confidence of detections to use
        """

        self.model = cv2.dnn.readNetFromONNX(model_path)
        self.labels = labels
        self.image_size = image_size
        self.min_confidence = min_confidence

    def detect(self, image: MatLike) -> list[Detection]:
        """Detect and classify items on an image.

        Args:
            image (MatLike): Image

        Returns:
            list[ItemDetection]: Detections
        """

        self.model.setInput(image)
        outputs = self.model.forward()
        outputs = np.array([cv2.transpose(outputs[0])])[0]

        boxes = []
        scores = []
        class_ids = []
        for output in outputs:
            class_scores = output[4:]
            (_, max_score, _, (_, max_class_id)) = cv2.minMaxLoc(class_scores)

            box = [
                output[0] - (0.5 * output[2]),
                output[1] - (0.5 * output[3]),
                output[2],
                output[3],
            ]
            boxes.append(box)
            scores.append(max_score)
            class_ids.append(max_class_id)

        box_indexes = cv2.dnn.NMSBoxes(boxes, scores, self.min_confidence, 0.5, 0.5)

        def to_detection(idx):
            class_name = self.labels[class_ids[idx]]
            confidence = scores[idx]
            box = utils.BoundingBox.from_xyhw_array(boxes[idx])
            return Detection(class_name, confidence, box)

        return list(map(to_detection, box_indexes))


class HandDetector:
    """Detects hands in an image."""

    def detect(self) -> list[Detection]:
        """Detect hands in an image.

        Returns:
            list[Detection]: Detections
        """

        hands: list[Detection] = []
        confidence = 1
        box = utils.BoundingBox.from_xyhw_array(np.array([300, 300, 150, 150]))
        pred_result = Detection("hand", confidence, box)
        hands.append(pred_result)

        return hands


class ItemDetector(Detector):
    """Detects and classifies product items in an image."""

    def __init__(
        self,
        model_path: str,
        labels: Mapping[int, str],
        image_size: utils.Size,
        min_confidence: float,
        max_hand_distance: float,
    ) -> None:
        """Detect and classifies product items in an image.

        Args:
            model_path (str): Path to model file
            labels (Mapping[int, str]): Class-to-label map
            image_size (utils.ImageSize): Size of image to detect from
            min_confidence (float): Minimum confidence of detections to use
            max_hand_distance (float): Max distance from hands to detect items
        """

        super().__init__(model_path, labels, image_size, min_confidence)
        self.max_hand_distance = max_hand_distance

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any], image_size: utils.Size
    ) -> ItemDetector:
        with open(config["label_path"], "r") as file:
            labels = yaml.safe_load(file)
        return cls(
            config["model_path"],
            labels,
            image_size,
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

        return list(
            filter(
                lambda i: any(
                    i.box.calcDistance(mask) < self.max_hand_distance
                    for mask in mask_boxes
                ),
                detections,
            )
        )
