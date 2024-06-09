from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import movinets
import movinets.config
import torch
from cv2.typing import MatLike
from torchvision.transforms import v2 as transforms

from jwo_cv.utils import Config

# Model config reference: https://github.com/Atze00/MoViNet-pytorch
MODEL_CONFIG = movinets.config._C.MODEL.MoViNetA2
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
        max_buffer_frame_count: int,
        min_detection_frame_span: int,
        device: str,
    ) -> None:
        """Detects and classifies actions in video using Movinet.

        Args:
            model (movinets.MoViNet): Model
            min_confidence (float): Minimum detection confidence
            max_buffer_frame_count (int): Num of video frames to buffer for detection
            max_detection_frame_span (int): Min num of frames between detection
            device (str): Device to run model on
        """

        self.min_confidence = min_confidence
        self.max_buffer_frame_count = max_buffer_frame_count
        self.min_detection_frame_span = min_detection_frame_span
        self.device = device

        self.model = model.to(self.device).eval()

        self.image_transforms = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Resize(size=IMAGE_SIZE),
            ]
        )

        self.buffer_frame_count = 0
        self.frame_count_since_last_detection = 0

    @classmethod
    def from_config(cls, config: Config, device: str) -> ActionClassifier:
        model = movinets.MoViNet(MODEL_CONFIG, causal=True, pretrained=True)

        model_weights = torch.load(config["model_path"])
        model.load_state_dict(model_weights)

        return ActionClassifier(
            model,
            config["min_confidence"],
            config["max_buffer_frame_count"],
            config["min_detection_frame_span"],
            device,
        )

    def clean_buffer_if_exceeds_duration(self):
        self.buffer_frame_count += 1

        if self.buffer_frame_count > self.max_buffer_frame_count:
            self.buffer_frame_count = 0
            self.model.clean_activation_buffers()

    def detect(self, image: MatLike) -> Action | None:
        """Detect pick or return action in a video frame.

        Args:
            image (MatLike): Image

        Returns:
            Action | None: Action or None if no action is detected.
        """

        self.clean_buffer_if_exceeds_duration()
        self.frame_count_since_last_detection += 1

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

        if (
            confidence >= self.min_confidence
            and self.frame_count_since_last_detection >= self.min_detection_frame_span
        ):
            self.frame_count_since_last_detection = 0
            self.model.clean_activation_buffers()
            logger.debug(
                "Action %s detected with confidence %f", action_type, confidence
            )
            return Action(class_id, action_type, confidence)
        return None
