from __future__ import annotations

from dataclasses import dataclass

import movinets
import movinets.config
import torch
from cv2.typing import MatLike
from torchvision.transforms import v2 as transforms

from jwo_cv import shop_event, utils
from jwo_cv.utils import Config

# Model config reference: https://github.com/Atze00/MoViNet-pytorch
MODEL_CONFIG = movinets.config._C.MODEL.MoViNetA2
IMAGE_SIZE = (172, 172)

PICK_CLASS_ID = 0
RETURN_CLASS_ID = 1

logger = utils.get_multiprocess_logger()


@dataclass(frozen=True)
class Action:
    """Describes an action recognition with class ID, type and confidence."""

    type: shop_event.ActionType
    confidence: float

    def __str__(self) -> str:
        return (
            f"{{class_id: {self.class_id}, type: {self.type}, "
            f"confidence: {self.confidence}}}"
        )

    @property
    def class_id(self) -> int:
        return (
            PICK_CLASS_ID
            if self.type == shop_event.ActionType.PICK
            else RETURN_CLASS_ID
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
    def from_config(cls, config: Config) -> ActionRecognizer:
        model = movinets.MoViNet(MODEL_CONFIG, causal=True, pretrained=True)
        model_weights = torch.load(config["model_path"])
        model.load_state_dict(model_weights)

        device = utils.get_device()
        logger.debug("Run action recognition model on %s", device)

        return ActionRecognizer(
            model,
            config["min_confidence"],
            config["min_action_frame_span"],
            device,
        )

    def recognize(self, image: MatLike) -> tuple[tuple[Action, Action], Action | None]:
        """Recognize pick or return action from a video frame.

        Args:
            image (MatLike): Video frame

        Returns:
            tuple[Action | None, tuple[Action, Action]]: Actions and recognized action
        """

        self.frame_count_since_last_action += 1

        input: torch.Tensor = self.image_transforms(image).to(self.device)
        # Add frame and batch dimension
        input = input[None, :, None]

        output: torch.Tensor = self.model(input)[0]
        action_probs = output.softmax(0)

        pick_prob = action_probs[PICK_CLASS_ID].item()
        return_prob = action_probs[RETURN_CLASS_ID].item()

        pick_action = Action(shop_event.ActionType.PICK, pick_prob)
        return_action = Action(shop_event.ActionType.RETURN, return_prob)
        actions = (pick_action, return_action)

        if pick_prob >= return_prob:
            best_action = pick_action
        else:
            best_action = return_action

        if (
            best_action.confidence >= self.min_confidence
            and self.frame_count_since_last_action >= self.min_action_frame_span
        ):
            self.frame_count_since_last_action = 0
            self.model.clean_activation_buffers()
            return actions, best_action

        return actions, None
