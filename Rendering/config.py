"""
Configuration module for foveated rendering.

Defines rendering parameters for progressive foveated rendering
with seamless transitions.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class RenderConfig:
    """
    Configuration for progressive foveated rendering.
    
    Attributes:
        window_width: Width of the render window in pixels.
        window_height: Height of the render window in pixels.
        fovea_radius: Radius of the fully-rendered central region.
        background_color: RGB color for non-rendered pixels.
        enable_gradients: Whether to enable smooth opacity fade at edges.
        degradation_rate: How quickly pixel density decreases with distance.
                         Lower = slower degradation, Higher = faster.
    """
    window_width: int = 1280
    window_height: int = 720
    fovea_radius: int = 50  # 100px diameter = 50px radius
    background_color: Tuple[int, int, int] = (0, 0, 0)
    enable_gradients: bool = True
    degradation_rate: float = 0.3  # Transition step as fraction of fovea_radius
    
    def __post_init__(self):
        if self.window_width <= 0 or self.window_height <= 0:
            raise ValueError("Window dimensions must be positive")
        if self.fovea_radius <= 0:
            raise ValueError("Fovea radius must be positive")
        if not 0.1 <= self.degradation_rate <= 1.0:
            raise ValueError("Degradation rate must be between 0.1 and 1.0")


def create_default_config() -> RenderConfig:
    """Factory function to create a default render configuration."""
    return RenderConfig()


def create_config_for_resolution(width: int, height: int) -> RenderConfig:
    """
    Factory function to create a configuration optimized for a given resolution.
    
    Args:
        width: Window width in pixels.
        height: Window height in pixels.
        
    Returns:
        RenderConfig optimized for the given resolution.
    """
    # Scale fovea radius based on screen size
    base_fovea = 50
    scale_factor = min(width, height) / 720
    scaled_fovea = int(base_fovea * scale_factor)
    
    return RenderConfig(
        window_width=width,
        window_height=height,
        fovea_radius=max(scaled_fovea, 30)  # Minimum 30px radius
    )
