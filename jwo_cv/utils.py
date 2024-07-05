from __future__ import annotations

import logging
import math
import multiprocessing
import os
import sys
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from numpy import typing as np_types

Config = Mapping[str, Any]
DEBUG_ENV_VAR = "JWO_CV_DEBUG"


class AppException(Exception):
    """Application-specific exception."""


@dataclass(frozen=True)
class Size:
    """Describes the (width, height) size of something."""

    width: int
    height: int

    def __str__(self) -> str:
        return f"({self.width}, {self.height})"

    @classmethod
    def from_wh_arr(cls, arr: np_types.NDArray | torch.Tensor | Sequence[int]) -> Size:
        return Size(int(arr[0]), int(arr[1]))

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
        self.bot_right = bot_right

        center_x = bot_right.x - (bot_right.x - top_left.x) / 2
        center_y = bot_right.y - (bot_right.y - top_left.y) / 2
        self.center = Position(round(center_x), round(center_y))

    def __str__(self) -> str:
        return (
            f"{{top_left: {self.top_left}, bot_right: {self.bot_right}, "
            f"center: {self.center}}}"
        )

    @classmethod
    def from_xyxy_arr(
        cls, array: np_types.NDArray | torch.Tensor | Sequence[int]
    ) -> BoundingBox:
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

    def calc_distance(self, box: BoundingBox) -> float:
        return math.dist(self.center.to_xy_arr(), box.center.to_xy_arr())


def get_device() -> str:
    """Get device to run vision ML models on."""

    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def get_log_handlers() -> list[logging.Handler]:
    """Get standard log handlers.

    Returns:
        list[logging.Handler]: Log handler
    """

    formatter = logging.Formatter(
        "[%(asctime)s|%(levelname)s|%(processName)s|%(name)s] %(message)s"
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    return [handler]


def get_multiprocess_logger() -> logging.Logger:
    """Get a logger which supports multiprocessing.
    Reference: https://stackoverflow.com/questions/641420/how-should-i-log-while-using-multiprocessing-in-python

    Returns:
        logging.Logger: Logger
    """

    logger = multiprocessing.get_logger()

    if os.getenv(DEBUG_ENV_VAR) == "1":
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    if not len(logger.handlers):
        for handler in get_log_handlers():
            logger.addHandler(handler)

    return logger
