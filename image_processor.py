"""
image_processor.py

Handles all OpenCV image loading, cloning, and programmatic difference
generation for the Spot-the-Difference game.

Classes:
    DifferenceRegion  - stores one difference's location and found-status
    ImageAlteration   - abstract base class for all alteration types
    ColourShiftAlt    - shifts hue/saturation (concrete alteration)
    BrightnessAlt     - changes local brightness (concrete alteration)
    PixelSwapAlt      - mirrors a patch horizontally (concrete alteration)
    TextureNoiseAlt   - adds structured noise (concrete alteration)
    ContrastAlt       - scales local contrast (concrete alteration)
    ImageProcessor    - loads image, creates clone, applies 5 differences
"""

import cv2
import numpy as np
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple



# DifferenceRegion — data class for one hidden difference


@dataclass
class DifferenceRegion:
    """
    Stores the bounding box (x, y, w, h) of one difference region,
    the alteration type applied, and whether the player has found it.
    """
    x: int               # top-left x coordinate
    y: int               # top-left y coordinate
    w: int               # width of the region
    h: int               # height of the region
    alteration_name: str = ""
    found: bool = False  # True once the player clicks it

    @property
    def centre(self) -> Tuple[int, int]:
        """Return the centre point of the region."""
        return self.x + self.w // 2, self.y + self.h // 2

    @property
    def radius(self) -> int:
        """Circle radius large enough to enclose the region."""
        return max(self.w, self.h) // 2 + 10

    def contains_point(self, px: int, py: int, tolerance: int = 30) -> bool:
        """
        Return True if the point (px, py) falls within this region
        plus a tolerance buffer so clicks don't have to be pixel-perfect.
        """
        cx, cy = self.centre
        return (abs(px - cx) <= self.w // 2 + tolerance and
                abs(py - cy) <= self.h // 2 + tolerance)


# ImageAlteration — abstract base class (demonstrates inheritance)


class ImageAlteration(ABC):
    """
    Abstract base class defining the interface every alteration must follow.
    Subclasses must implement:
        name  (property) : human-readable label
        apply (method)   : modify the image ROI and return the image
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this alteration type."""

    @abstractmethod
    def apply(self, image: np.ndarray, x: int, y: int,
              w: int, h: int) -> np.ndarray:
        """
        Apply the alteration to *image* inside the rectangle (x, y, w, h).
        Modifies in-place and returns the modified image.
        """



# Concrete alteration types — all inherit from ImageAlteration (polymorphism)


class ColourShiftAlt(ImageAlteration):
    """
    Shifts the hue and boosts saturation of a rectangular region
    using OpenCV's HSV colour space.
    The change is noticeable on careful inspection but not glaringly obvious.
    """

    def __init__(self, hue_delta: int = 20, sat_scale: float = 1.4):
        self._hue_delta = hue_delta    # how much to rotate the hue (0-180)
        self._sat_scale = sat_scale    # saturation multiplier

    @property
    def name(self) -> str:
        return "ColourShift"

    def apply(self, image: np.ndarray, x: int, y: int,
              w: int, h: int) -> np.ndarray:
        roi = image[y:y + h, x:x + w].copy()
        # Convert to HSV, shift hue, scale saturation, convert back to BGR
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.int32)
        hsv[:, :, 0] = (hsv[:, :, 0] + self._hue_delta) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * self._sat_scale, 0, 255)
        hsv = hsv.astype(np.uint8)
        image[y:y + h, x:x + w] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return image


class BrightnessAlt(ImageAlteration):
    """
    Increases or decreases the brightness of a rectangular region
    by adding a fixed offset to all pixel values.
    """

    def __init__(self, delta: int = 55):
        self._delta = delta   # positive = brighter, negative = darker

    @property
    def name(self) -> str:
        return "Brightness"

    def apply(self, image: np.ndarray, x: int, y: int,
              w: int, h: int) -> np.ndarray:
        roi = image[y:y + h, x:x + w].astype(np.int32)
        roi = np.clip(roi + self._delta, 0, 255).astype(np.uint8)
        image[y:y + h, x:x + w] = roi
        return image


class PixelSwapAlt(ImageAlteration):
    """
    Horizontally flips (mirrors) a rectangular patch.
    Creates a subtle mirror-image artefact as the difference.
    """

    @property
    def name(self) -> str:
        return "PixelSwap"

    def apply(self, image: np.ndarray, x: int, y: int,
              w: int, h: int) -> np.ndarray:
        roi = image[y:y + h, x:x + w].copy()
        image[y:y + h, x:x + w] = cv2.flip(roi, 1)   # 1 = horizontal flip
        return image


class TextureNoiseAlt(ImageAlteration):
    """
    Overlays a structured random noise pattern on a region,
    creating a subtle grainy texture difference.
    """

    def __init__(self, intensity: int = 40):
        self._intensity = intensity   # max pixel offset per channel

    @property
    def name(self) -> str:
        return "TextureNoise"

    def apply(self, image: np.ndarray, x: int, y: int,
              w: int, h: int) -> np.ndarray:
        noise = np.random.randint(-self._intensity, self._intensity,
                                   (h, w, 3), dtype=np.int32)
        roi = image[y:y + h, x:x + w].astype(np.int32)
        image[y:y + h, x:x + w] = np.clip(roi + noise, 0, 255).astype(np.uint8)
        return image


class ContrastAlt(ImageAlteration):
    """
    Reduces local contrast by scaling pixel values toward the region mean,
    producing a washed-out or flattened look.
    """

    def __init__(self, alpha: float = 0.55):
        self._alpha = alpha   # <1 reduces contrast, >1 increases it

    @property
    def name(self) -> str:
        return "Contrast"

    def apply(self, image: np.ndarray, x: int, y: int,
              w: int, h: int) -> np.ndarray:
        roi = image[y:y + h, x:x + w].astype(np.float32)
        mean = roi.mean()
        adjusted = np.clip((roi - mean) * self._alpha + mean, 0, 255)
        image[y:y + h, x:x + w] = adjusted.astype(np.uint8)
        return image


# ImageProcessor — orchestrates loading and difference generation


class ImageProcessor:
    """
    Loads an image file, resizes it for display, creates an exact clone,
    then programmatically introduces exactly 5 non-overlapping differences
    into the clone using randomly selected alteration types.

    Demonstrates:
        - Encapsulation  : all image data and logic kept inside this class
        - Class interaction : uses DifferenceRegion and ImageAlteration objects
        - Polymorphism   : calls alt.apply() on any ImageAlteration subclass

    Attributes (available after load_image):
        original_bgr  (np.ndarray)          original image array
        modified_bgr  (np.ndarray)          cloned image with 5 differences
        differences   (List[DifferenceRegion])  the 5 difference locations
    """

    NUM_DIFFERENCES: int = 5
    MIN_REGION_SIZE: int = 40    # minimum side length in pixels
    MAX_REGION_SIZE: int = 90    # maximum side length in pixels

    def __init__(self) -> None:
        # All available alteration objects — more than 5 so we can randomise
        self._alteration_pool: List[ImageAlteration] = [
            ColourShiftAlt(hue_delta=25),
            ColourShiftAlt(hue_delta=-22),
            BrightnessAlt(delta=60),
            BrightnessAlt(delta=-60),
            PixelSwapAlt(),
            TextureNoiseAlt(intensity=45),
            ContrastAlt(alpha=0.50),
        ]

        # These are set when load_image() is called
        self.original_bgr: np.ndarray = None
        self.modified_bgr: np.ndarray = None
        self.differences: List[DifferenceRegion] = []
        self._img_w: int = 0
        self._img_h: int = 0


    # Public methods
   
    def load_image(self, path: str,
                   display_size: Tuple[int, int] = (550, 430)) -> bool:
        """
        Load the image at *path*, resize it to *display_size* (width, height),
        clone it, and apply 5 random non-overlapping differences.
        Returns True on success, False if the file cannot be opened.
        """
        raw = cv2.imread(path)
        if raw is None:
            return False    # file not found or unsupported format

        dw, dh = display_size
        self.original_bgr = cv2.resize(raw, (dw, dh),
                                        interpolation=cv2.INTER_AREA)
        self._img_w, self._img_h = dw, dh

        # Clone the original — this copy will receive the alterations
        self.modified_bgr = self.original_bgr.copy()
        self.differences = []

        self._apply_differences()
        return True

    def draw_circle(self, image: np.ndarray, region: DifferenceRegion,
                    colour: Tuple[int, int, int],
                    thickness: int = 3) -> np.ndarray:
        """Draw an anti-aliased circle centred on *region* onto *image*."""
        cx, cy = region.centre
        cv2.circle(image, (cx, cy), region.radius,
                   colour, thickness, lineType=cv2.LINE_AA)
        return image

    def get_original_with_overlays(
            self,
            found_regions: List[DifferenceRegion],
            revealed_regions: List[DifferenceRegion]) -> np.ndarray:
        """
        Return a copy of the original image with circles drawn:
          - RED   around differences the player has already found
          - BLUE  around differences that were revealed via the Reveal button
        """
        img = self.original_bgr.copy()
        for r in found_regions:
            self.draw_circle(img, r, (0, 0, 220))      # BGR → red
        for r in revealed_regions:
            self.draw_circle(img, r, (220, 100, 0))    # BGR → blue
        return img

    def get_modified_with_overlays(
            self,
            found_regions: List[DifferenceRegion],
            revealed_regions: List[DifferenceRegion]) -> np.ndarray:
        """Same as get_original_with_overlays but for the modified image."""
        img = self.modified_bgr.copy()
        for r in found_regions:
            self.draw_circle(img, r, (0, 0, 220))
        for r in revealed_regions:
            self.draw_circle(img, r, (220, 100, 0))
        return img


    # Private helpers


    def _apply_differences(self) -> None:
        """
        Randomly pick 5 distinct alterations from the pool, find a valid
        non-overlapping position for each, apply them to the clone, and
        store the resulting DifferenceRegion objects in self.differences.
        """
        random.shuffle(self._alteration_pool)
        chosen = random.sample(self._alteration_pool, self.NUM_DIFFERENCES)

        placed: List[DifferenceRegion] = []

        for alt in chosen:
            region = self._find_non_overlapping_region(placed)
            self.modified_bgr = alt.apply(
                self.modified_bgr,
                region.x, region.y, region.w, region.h
            )
            region.alteration_name = alt.name
            placed.append(region)

        self.differences = placed

    def _find_non_overlapping_region(
            self, placed: List[DifferenceRegion],
            max_attempts: int = 500) -> DifferenceRegion:
        """
        Randomly generate rectangles until one is found that does not
        overlap any region already in *placed* (with a 20-pixel buffer).
        If all attempts fail, return a fallback position.
        """
        for _ in range(max_attempts):
            w = random.randint(self.MIN_REGION_SIZE, self.MAX_REGION_SIZE)
            h = random.randint(self.MIN_REGION_SIZE, self.MAX_REGION_SIZE)
            x = random.randint(0, self._img_w - w)
            y = random.randint(0, self._img_h - h)
            candidate = DifferenceRegion(x=x, y=y, w=w, h=h)
            if not self._overlaps_any(candidate, placed):
                return candidate

        # Very rare fallback
        return DifferenceRegion(
            x=random.randint(0, self._img_w - self.MIN_REGION_SIZE),
            y=random.randint(0, self._img_h - self.MIN_REGION_SIZE),
            w=self.MIN_REGION_SIZE,
            h=self.MIN_REGION_SIZE,
        )

    @staticmethod
    def _overlaps_any(candidate: DifferenceRegion,
                      placed: List[DifferenceRegion],
                      padding: int = 20) -> bool:
        """
        Return True if *candidate* intersects any region in *placed*
        (expanded by *padding* pixels on all sides to ensure clear spacing).
        """
        cx1, cy1 = candidate.x - padding, candidate.y - padding
        cx2, cy2 = (candidate.x + candidate.w + padding,
                    candidate.y + candidate.h + padding)
        for p in placed:
            if cx1 < p.x + p.w and cx2 > p.x and cy1 < p.y + p.h and cy2 > p.y:
                return True
        return False
