from dataclasses import dataclass
from cv2.typing import MatLike
from ultralytics import YOLO
from jwo_cv import utils


@dataclass(frozen=True)
class ItemPrediction:
    class_name: str
    confidence: float
    box: utils.BoundingBox

    def __str__(self) -> str:
        return f"{{Class: {self.class_name}, Confidence: {self.confidence}, Box: {self.box}}}"


class ItemClassifier:
    def __init__(self, model_path: str) -> None:
        self.model = YOLO(model_path)

    def predict(self, image: MatLike) -> list[ItemPrediction]:
        results = self.model.predict(image, verbose=False)

        items: list[ItemPrediction] = []
        for result in results:
            for box in result.boxes:
                class_name = self.model.names[int(box.cls)]
                confidence = float(box.conf)
                xyxy = box.xyxy[0]
                bounding_box = utils.BoundingBox(
                    utils.Position(xyxy[0], xyxy[1]),
                    utils.Position(xyxy[2], xyxy[3]),
                )
                pred_result = ItemPrediction(class_name, confidence, bounding_box)
                items.append(pred_result)

        return items
