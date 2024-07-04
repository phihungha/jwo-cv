from __future__ import annotations

import logging
from dataclasses import dataclass

import movinets
import movinets.config
import torch
from cv2.typing import MatLike
from torchvision.transforms import v2 as transforms

from jwo_cv import shop_event
from jwo_cv.utils import Config

# Model config reference: https://github.com/Atze00/MoViNet-pytorch
MODEL_CONFIG = movinets.config._C.MODEL.MoViNetA2
IMAGE_SIZE = (224, 224)

PICK_CLASS_ID = 0
RETURN_CLASS_ID = 1

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Action:
    """Describes an action recognition with class ID, type and confidence."""

    class_id: int
    type: shop_event.ActionType
    confidence: float

    def __str__(self) -> str:
        return (
            f"{{class_id: {self.class_id}, type: {self.type}, "
            f"confidence: {self.confidence}}}"
        )


class ActionRecognizer:
    """Recognizes action in video using Movinet."""

    def __init__(
        self,
        model: movinets.MoViNet,
        min_confidence: float,
        min_detection_frame_span: int,
        device: str,
    ) -> None:
        """Recognizes action in video using Movinet.

        Args:
            model (movinets.MoViNet): Model
            min_confidence (float): Minimum detection confidence
            max_detection_frame_span (int): Min num of frames between detection
            device (str): Device to run model on
        """

        self.min_confidence = min_confidence
        self.min_action_frame_span = min_detection_frame_span
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
        self.frame_count_since_last_action = 0

    @classmethod
    def from_config(cls, config: Config, device: str) -> ActionRecognizer:
        model = movinets.MoViNet(MODEL_CONFIG, causal=True, pretrained=True)

        model_weights = torch.load(config["model_path"])
        model.load_state_dict(model_weights)

        return ActionRecognizer(
            model,
            config["min_confidence"],
            config["min_action_frame_span"],
            device,
        )

    def recognize(self, image: MatLike) -> Action | None:
        """Recognize pick or return action from a video frame.

        Args:
            image (MatLike): Image

        Returns:
            tuple[Action, float] | None: Action or None if no action is detected.
        """

        self.frame_count_since_last_action += 1

        input: torch.Tensor = self.image_transforms(image).to(self.device)
        # Add frame and batch dimension
        input = input[None, :, None]

        output: torch.Tensor = self.model(input)[0]
        action_probs = output.softmax(0)

        pick_prob = action_probs[PICK_CLASS_ID].item()
        return_prob = action_probs[RETURN_CLASS_ID].item()

        if pick_prob >= return_prob:
            class_id = PICK_CLASS_ID
            action_type = shop_event.ActionType.PICK
            confidence = pick_prob
        else:
            class_id = RETURN_CLASS_ID
            action_type = shop_event.ActionType.RETURN
            confidence = return_prob

        if (
            confidence >= self.min_confidence
            and self.frame_count_since_last_action >= self.min_action_frame_span
        ):
            self.frame_count_since_last_action = 0
            self.model.clean_activation_buffers()
            return Action(class_id, action_type, confidence)

        return None
