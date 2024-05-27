from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import movinets
import movinets.config
import torch
from cv2.typing import MatLike
from torchvision.transforms import v2 as transforms

from jwo_cv.utils import Config

# Model config reference: https://github.com/Atze00/MoViNet-pytorch
MODEL_CONFIG = movinets.config._C.MODEL.MoViNetA2
MODEL_WEIGHT_PATH = "jwo_cv/config/action_movineta2.pth"
IMAGE_SIZE = (224, 224)

PICK_CLASS_ID = 0
RETURN_CLASS_ID = 1

logger = logging.getLogger(__name__)


class ActionType(Enum):
    PICK = 0
    RETURN = 1


@dataclass(frozen=True)
class Action:
    """Describes an action detection with class ID, type and confidence."""

    class_id: int
    type: ActionType
    confidence: float

    def __str__(self) -> str:
        return (
            f"{{class_id: {self.class_id}, type: {self.type}, "
            f"confidence: {self.confidence}}}"
        )


class ActionClassifier:
    """Detects and classifies actions in video using Movinet."""

    def __init__(
        self,
        model: movinets.MoViNet,
        min_confidence: float,
        buffer_duration: int,
        device: str,
    ) -> None:
        """Detects and classifies actions in video using Movinet.

        Args:
            model (movinets.MoViNet): Model
            min_confidence (float): Minimum detection confidence
            buffer_duration (int): Seconds of video frames to buffer for detection
            device (str): Device to run model on
        """

        self.min_confidence = min_confidence
        self.buffer_duration = buffer_duration
        self.device = device

        self.model = model.to(self.device).eval()

        self.image_transforms = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Resize(size=IMAGE_SIZE),
            ]
        )

        self.last_detection_time = time.time()

    @classmethod
    def from_config(cls, config: Config, device: str) -> ActionClassifier:
        model = movinets.MoViNet(MODEL_CONFIG, causal=True, pretrained=True)
        model_weights = torch.load(MODEL_WEIGHT_PATH)
        model.load_state_dict(model_weights)
        return ActionClassifier(
            model, config["min_confidence"], config["stream_buffer_duration"], device
        )

    def clean_buffer_if_exceeds_duration(self):
        current_time = time.time()
        if current_time > self.last_detection_time + self.buffer_duration:
            self.last_detection_time = current_time
            self.model.clean_activation_buffers()
            logger.debug("Reset action stream buffer")

    def detect(self, image: MatLike) -> Action | None:
        """Detect pick or return action in a video frame.

        Args:
            image (MatLike): Image

        Returns:
            Action | None: Action or None if no action is detected.
        """

        self.clean_buffer_if_exceeds_duration()

        input: torch.Tensor = self.image_transforms(image).to(self.device)
        # Add frame and batch dimension
        input = input.unsqueeze(1).unsqueeze(0)

        output: torch.Tensor = self.model(input)[0]
        action_probs = output.softmax(0)

        pick_prob = action_probs[PICK_CLASS_ID].item()
        return_prob = action_probs[RETURN_CLASS_ID].item()

        if pick_prob >= return_prob:
            class_id = PICK_CLASS_ID
            action_type = ActionType.PICK
            confidence = pick_prob
        else:
            class_id = RETURN_CLASS_ID
            action_type = ActionType.RETURN
            confidence = return_prob

        if confidence >= self.min_confidence:
            self.model.clean_activation_buffers()
            logger.info(
                "%s: Action %s detected with confidence %f",
                datetime.now(),
                action_type,
                confidence,
            )
            return Action(class_id, action_type, confidence)
        return None
