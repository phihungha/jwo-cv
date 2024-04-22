from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy import typing as np_types


@dataclass(frozen=True)
class Size:
    """Describes the (width, height) size pf something."""

    width: int
    height: int

    def __str__(self) -> str:
        return f"({self.width}, {self.height})"

    @classmethod
    def from_wh_arr(cls, arr: np_types.NDArray | Sequence[int]) -> Size:
        return Size(arr[0], arr[1])

    def to_wh_arr(self) -> np_types.NDArray:
        return np.array([self.width, self.height])


@dataclass(frozen=True)
class Position:
    """Describes a (x, y) position of something."""

    x: int
    y: int

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

    def to_xy_arr(self) -> np_types.NDArray:
        return np.array([self.x, self.y])


class BoundingBox:
    """Describes the bounding box of an object detection."""

    def __init__(self, top_left: Position, bot_right: Position) -> None:
        """Describes the bounding box of an object detection.

        Args:
            top_left (Point): Top-left corner position
            bottom_right (Point): Bottom-right corner position
        """

        self.top_left = top_left
        self.top_right = Position(bot_right.x, top_left.y)
        self.bot_left = Position(top_left.x, bot_right.y)
        self.bot_right = bot_right

    def __str__(self) -> str:
        return f"{{top_left: {self.top_left}, bot_right: {self.bot_right}}}"

    @classmethod
    def from_xyxy_arr(cls, array: np_types.NDArray | Sequence[int]) -> BoundingBox:
        return cls(
            Position(int(array[0]), int(array[1])),
            Position(int(array[2]), int(array[3])),
        )

    def to_xyxy_arr(self) -> np_types.NDArray:
        return np.array(
            [
                self.top_left.x,
                self.top_left.y,
                self.bot_right.x,
                self.bot_right.y,
            ]
        )

    def check_overlap_box(self, box: BoundingBox) -> bool:
        overlaps_horizontal = (
            self.top_left.x <= box.top_left.x <= self.top_right.x
            or self.top_left.x <= box.top_right.x <= self.top_right.x
        )
        overlaps_vertical = (
            self.top_left.y <= box.top_left.y <= self.bot_left.y
            or self.top_left.y <= box.bot_left.y <= self.bot_left.y
        )
        return overlaps_horizontal and overlaps_vertical
