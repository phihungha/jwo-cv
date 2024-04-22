from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import logging
import time

import movinets.config
import torch
from torchvision.transforms import v2 as transforms
from cv2.typing import MatLike
import movinets

from jwo_cv import utils

# https://github.com/Atze00/MoViNet-pytorch
MODEL_CONFIG = movinets.config._C.MODEL.MoViNetA2
IMAGE_SIZE = (224, 224)
PICK_CLASS_ID = 581
RETURN_CLASS_ID = 582

logger = logging.getLogger(__name__)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class ActionType(Enum):
    PICK = 0
    RETURN = 1


@dataclass(frozen=True)
class Action:
    """Describes an action with type and confidence."""

    type: ActionType
    confidence: float


class ActionClassifier:
    def __init__(
        self,
        min_confidence: float,
        stream_buffer_duration_sec: int,
    ) -> None:
        self.min_confidence = min_confidence
        self.stream_buffer_duration_sec = stream_buffer_duration_sec

        self.model = movinets.MoViNet(MODEL_CONFIG, causal=True, pretrained=True)
        self.model = self.model.to(device)
        self.model.eval()

        self.image_transforms = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Resize(size=IMAGE_SIZE),
            ]
        )

        self.last_prediction_time = time.time()

    @classmethod
    def from_config(cls, config: utils.Config) -> ActionClassifier:
        return ActionClassifier(
            config["min_confidence"], config["stream_buffer_duration_sec"]
        )

    def predict(self, image: MatLike) -> Action | None:
        current_time = time.time()
        if current_time > self.last_prediction_time + self.stream_buffer_duration_sec:
            self.last_prediction_time = current_time
            self.model.clean_activation_buffers()
            logger.debug("Reset buffer.")

        input: torch.Tensor = self.image_transforms(image).to(device)
        # Add frame dimension
        input = input.unsqueeze(1)
        # Add batch dimension
        input = input.unsqueeze(0)

        output: torch.Tensor = self.model(input)[0]
        probabilities = output.softmax(0)

        pick_prob = probabilities[PICK_CLASS_ID].item()
        return_prob = probabilities[RETURN_CLASS_ID].item()

        if pick_prob >= return_prob:
            action_type = ActionType.PICK
            confidence = pick_prob
        else:
            action_type = ActionType.RETURN
            confidence = return_prob

        if confidence >= self.min_confidence:
            return Action(action_type, confidence)
        return None
