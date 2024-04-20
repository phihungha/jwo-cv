from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Sequence

from numpy import typing as np_types


@dataclass(frozen=True)
class Size:
    """Describes the width and height of an object."""

    width: int
    height: int

    @classmethod
    def from_wh_seq(cls, seq: Sequence[int]) -> Size:
        return Size(seq[0], seq[1])

    def to_wh_seq(self) -> Sequence[int]:
        return [self.width, self.height]


@dataclass(frozen=True)
class Position:
    """Describes a (x, y) position by pixel on an image."""

    x: int
    y: int

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

    def to_xy_list(self) -> list[int]:
        return [self.x, self.y]


class BoundingBox:
    """Describes the bounding box of an object detection."""

    def __init__(self, center: Position, size: Size) -> None:
        """Describes the bounding box of an object detection.

        Args:
            top_left (Point): Top-left corner position
            bottom_right (Point): Bottom-right corner position
        """

        self.center = center
        self.size = size

        top_left_x = round(center.x - size.width / 2)
        top_left_y = round(center.y - size.height / 2)
        self.top_left = Position(top_left_x, top_left_y)

        bot_right_x = top_left_x + size.width
        bot_right_y = top_left_y + size.height
        self.bot_right = Position(bot_right_x, bot_right_y)

    def __str__(self) -> str:
        return f"{{top_left: {self.center}, bottom_right: {self.size}, center: {self.center}}}"

    @classmethod
    def from_xyhw_array(cls, array: np_types.NDArray) -> BoundingBox:
        return cls(
            Position(round(array[0]), round(array[1])),
            Size(round(array[2]), round(array[3])),
        )

    def to_xyxy_list(self) -> list[int]:
        return [
            self.top_left.x,
            self.top_left.y,
            self.bot_right.x,
            self.bot_right.y,
        ]

    def calcDistance(self, box: BoundingBox) -> float:
        return math.dist(self.center.to_xy_list(), box.center.to_xy_list())
