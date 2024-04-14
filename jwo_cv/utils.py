from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Position:
    x: int
    y: int

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"


@dataclass(frozen=True)
class BoundingBox:
    top_left: Position
    bottom_right: Position

    def __str__(self) -> str:
        return f"{{top_left: {self.top_left}, lower_right: {self.bottom_right}}}"

    @property
    def tensor(self) -> torch.Tensor:
        return torch.Tensor(
            [self.top_left.x, self.top_left.y, self.bottom_right.x, self.bottom_right.y]
        )


def calcBoundingBoxDistance(box1: BoundingBox, box2: BoundingBox) -> int:
    return box2.top_left.x - box1.top_left.x
