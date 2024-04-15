from __future__ import annotations
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class Position:
    """Describes a (x, y) position on an image."""

    x: int | float
    y: int | float

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

    def to_tensor(self) -> torch.Tensor:
        return torch.Tensor([self.x, self.y])


class BoundingBox:
    """Describes the bounding box of an object detection."""

    def __init__(self, top_left: Position, bottom_right: Position) -> None:
        """Describes the bounding box of an object detection.

        Args:
            top_left (Position): Top-left corner position
            bottom_right (Position): Bottom-right corner position
        """

        self.top_left = top_left
        self.bottom_right = bottom_right

        center_x = self.bottom_right.x - (self.bottom_right.x - self.top_left.x) / 2
        center_y = self.bottom_right.y - (self.bottom_right.y - self.top_left.y) / 2
        self.center = Position(round(center_x), round(center_y))

    def __str__(self) -> str:
        return f"{{top_left: {self.top_left}, bottom_right: {self.bottom_right}, center: {self.center}}}"

    @classmethod
    def from_xyxy_tensor(cls, xyxy: torch.Tensor) -> BoundingBox:
        return cls(
            Position(int(xyxy[0]), int(xyxy[1])),
            Position(int(xyxy[2]), int(xyxy[3])),
        )

    def to_tensor(self) -> torch.Tensor:
        return torch.Tensor(
            [self.top_left.x, self.top_left.y, self.bottom_right.x, self.bottom_right.y]
        )

    def calcDistance(self, box: BoundingBox) -> float:
        return math.dist(self.center.to_tensor(), box.center.to_tensor())
