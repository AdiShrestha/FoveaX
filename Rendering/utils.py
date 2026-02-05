"""
Utility functions for foveated rendering.

Provides helper functions for image processing, performance measurement,
and common operations.
"""

import numpy as np
from typing import Tuple, Optional, Callable
from functools import wraps
import time
import logging

logger = logging.getLogger(__name__)


class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str = "Operation", log: bool = True):
        """
        Initialize timer.
        
        Args:
            name: Name of the operation being timed.
            log: Whether to log the result.
        """
        self.name = name
        self.log = log
        self.elapsed: float = 0.0
    
    def __enter__(self) -> 'Timer':
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args) -> None:
        self.elapsed = time.perf_counter() - self.start_time
        if self.log:
            logger.debug(f"{self.name} took {self.elapsed * 1000:.2f}ms")


def timed(func: Callable) -> Callable:
    """
    Decorator to time function execution.
    
    Args:
        func: Function to wrap.
        
    Returns:
        Wrapped function that logs execution time.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        with Timer(func.__name__):
            return func(*args, **kwargs)
    return wrapper


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp a value to a range.
    
    Args:
        value: Value to clamp.
        min_val: Minimum allowed value.
        max_val: Maximum allowed value.
        
    Returns:
        Clamped value.
    """
    return max(min_val, min(value, max_val))


def lerp(a: float, b: float, t: float) -> float:
    """
    Linear interpolation between two values.
    
    Args:
        a: Start value.
        b: End value.
        t: Interpolation factor (0.0 to 1.0).
        
    Returns:
        Interpolated value.
    """
    return a + (b - a) * clamp(t, 0.0, 1.0)


def smooth_step(edge0: float, edge1: float, x: float) -> float:
    """
    Smooth Hermite interpolation.
    
    Args:
        edge0: Lower edge.
        edge1: Upper edge.
        x: Value to interpolate.
        
    Returns:
        Smoothly interpolated value.
    """
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def compute_euclidean_distance(
    x1: int, y1: int, x2: int, y2: int
) -> float:
    """
    Compute Euclidean distance between two points.
    
    Args:
        x1, y1: First point coordinates.
        x2, y2: Second point coordinates.
        
    Returns:
        Euclidean distance.
    """
    dx = x2 - x1
    dy = y2 - y1
    return np.sqrt(dx * dx + dy * dy)


def create_circular_mask(
    width: int, 
    height: int,
    center: Tuple[int, int],
    radius: float
) -> np.ndarray:
    """
    Create a circular boolean mask.
    
    Args:
        width: Mask width.
        height: Mask height.
        center: Center (x, y) of the circle.
        radius: Circle radius.
        
    Returns:
        Boolean numpy array where True = inside circle.
    """
    y, x = np.ogrid[:height, :width]
    cx, cy = center
    distance_sq = (x - cx) ** 2 + (y - cy) ** 2
    return distance_sq <= radius ** 2


def create_ring_mask(
    width: int,
    height: int,
    center: Tuple[int, int],
    inner_radius: float,
    outer_radius: float
) -> np.ndarray:
    """
    Create a ring (annulus) boolean mask.
    
    Args:
        width: Mask width.
        height: Mask height.
        center: Center (x, y) of the ring.
        inner_radius: Inner radius.
        outer_radius: Outer radius.
        
    Returns:
        Boolean numpy array where True = inside ring.
    """
    y, x = np.ogrid[:height, :width]
    cx, cy = center
    distance_sq = (x - cx) ** 2 + (y - cy) ** 2
    return (distance_sq > inner_radius ** 2) & (distance_sq <= outer_radius ** 2)


def apply_gaussian_blur(
    image: np.ndarray,
    sigma: float = 1.0
) -> np.ndarray:
    """
    Apply Gaussian blur to an image.
    
    Uses a simple convolution approach without external dependencies.
    
    Args:
        image: Input image array.
        sigma: Standard deviation of the Gaussian.
        
    Returns:
        Blurred image.
    """
    if sigma <= 0:
        return image.copy()
    
    # Create Gaussian kernel
    size = int(6 * sigma + 1)
    if size % 2 == 0:
        size += 1
    
    x = np.arange(size) - size // 2
    kernel_1d = np.exp(-x ** 2 / (2 * sigma ** 2))
    kernel_1d /= kernel_1d.sum()
    
    # Separable convolution
    result = image.astype(float)
    
    # Horizontal pass
    for c in range(result.shape[2] if len(result.shape) > 2 else 1):
        if len(result.shape) > 2:
            result[:, :, c] = np.apply_along_axis(
                lambda row: np.convolve(row, kernel_1d, mode='same'),
                axis=1,
                arr=result[:, :, c]
            )
        else:
            result = np.apply_along_axis(
                lambda row: np.convolve(row, kernel_1d, mode='same'),
                axis=1,
                arr=result
            )
    
    # Vertical pass
    for c in range(result.shape[2] if len(result.shape) > 2 else 1):
        if len(result.shape) > 2:
            result[:, :, c] = np.apply_along_axis(
                lambda col: np.convolve(col, kernel_1d, mode='same'),
                axis=0,
                arr=result[:, :, c]
            )
        else:
            result = np.apply_along_axis(
                lambda col: np.convolve(col, kernel_1d, mode='same'),
                axis=0,
                arr=result
            )
    
    return result.astype(image.dtype)


def blend_images(
    image1: np.ndarray,
    image2: np.ndarray,
    mask: np.ndarray,
    feather: int = 0
) -> np.ndarray:
    """
    Blend two images using a mask.
    
    Args:
        image1: First image (shown where mask is True).
        image2: Second image (shown where mask is False).
        mask: Boolean mask array.
        feather: Feather radius for smooth blending.
        
    Returns:
        Blended image.
    """
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions")
    
    result = image2.copy()
    
    if feather > 0:
        # Convert mask to float and blur for feathering
        mask_float = mask.astype(float)
        mask_float = apply_gaussian_blur(
            mask_float[:, :, np.newaxis] if len(mask_float.shape) == 2 else mask_float,
            sigma=feather / 3
        )
        if len(mask_float.shape) > 2:
            mask_float = mask_float[:, :, 0]
        
        # Blend using the feathered mask
        for c in range(image1.shape[2] if len(image1.shape) > 2 else 1):
            if len(image1.shape) > 2:
                result[:, :, c] = (
                    image1[:, :, c] * mask_float +
                    image2[:, :, c] * (1 - mask_float)
                ).astype(image1.dtype)
            else:
                result = (
                    image1 * mask_float +
                    image2 * (1 - mask_float)
                ).astype(image1.dtype)
    else:
        result[mask] = image1[mask]
    
    return result


def calculate_rendering_efficiency(
    total_pixels: int,
    rendered_pixels: int
) -> float:
    """
    Calculate rendering efficiency as a percentage.
    
    Args:
        total_pixels: Total number of pixels in the image.
        rendered_pixels: Number of pixels actually rendered.
        
    Returns:
        Efficiency percentage (pixels saved).
    """
    if total_pixels <= 0:
        return 0.0
    return 100.0 * (1.0 - rendered_pixels / total_pixels)


def validate_image_array(
    array: np.ndarray,
    expected_channels: Optional[int] = None
) -> bool:
    """
    Validate that an array is a valid image.
    
    Args:
        array: Array to validate.
        expected_channels: Expected number of channels (None for any).
        
    Returns:
        True if valid, False otherwise.
    """
    if not isinstance(array, np.ndarray):
        return False
    
    if len(array.shape) < 2:
        return False
    
    if len(array.shape) == 2:
        # Grayscale
        return expected_channels is None or expected_channels == 1
    
    if len(array.shape) == 3:
        channels = array.shape[2]
        if expected_channels is not None and channels != expected_channels:
            return False
        return channels in (1, 3, 4)
    
    return False


def resize_image(
    image: np.ndarray,
    new_width: int,
    new_height: int,
    interpolation: str = 'bilinear'
) -> np.ndarray:
    """
    Resize an image using numpy (no external dependencies).
    
    Args:
        image: Input image array.
        new_width: Target width.
        new_height: Target height.
        interpolation: Interpolation method ('nearest' or 'bilinear').
        
    Returns:
        Resized image.
    """
    old_height, old_width = image.shape[:2]
    
    # Create coordinate grids
    x_ratio = old_width / new_width
    y_ratio = old_height / new_height
    
    x_coords = np.arange(new_width) * x_ratio
    y_coords = np.arange(new_height) * y_ratio
    
    if interpolation == 'nearest':
        x_indices = np.floor(x_coords).astype(int)
        y_indices = np.floor(y_coords).astype(int)
        
        x_indices = np.clip(x_indices, 0, old_width - 1)
        y_indices = np.clip(y_indices, 0, old_height - 1)
        
        return image[y_indices[:, np.newaxis], x_indices]
    
    elif interpolation == 'bilinear':
        x0 = np.floor(x_coords).astype(int)
        y0 = np.floor(y_coords).astype(int)
        x1 = np.minimum(x0 + 1, old_width - 1)
        y1 = np.minimum(y0 + 1, old_height - 1)
        
        x_frac = x_coords - x0
        y_frac = y_coords - y0
        
        # Bilinear interpolation
        result = np.zeros((new_height, new_width) + image.shape[2:], dtype=image.dtype)
        
        for j in range(new_height):
            for i in range(new_width):
                top_left = image[y0[j], x0[i]]
                top_right = image[y0[j], x1[i]]
                bottom_left = image[y1[j], x0[i]]
                bottom_right = image[y1[j], x1[i]]
                
                top = top_left * (1 - x_frac[i]) + top_right * x_frac[i]
                bottom = bottom_left * (1 - x_frac[i]) + bottom_right * x_frac[i]
                result[j, i] = (top * (1 - y_frac[j]) + bottom * y_frac[j]).astype(image.dtype)
        
        return result
    
    else:
        raise ValueError(f"Unknown interpolation method: {interpolation}")
