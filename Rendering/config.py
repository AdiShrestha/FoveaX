"""
Configuration module for foveated rendering.

Defines rendering zones, degradation patterns, and configurable parameters.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
from enum import Enum, auto


class DegradationPattern(Enum):
    """Enumeration of pixel degradation patterns."""
    FULL = auto()           # All pixels rendered
    SPARSE_1 = auto()       # 2 rendered, 1 black (66% density)
    CHECKERBOARD = auto()   # 1 rendered, 1 black (50% density)
    SPARSE_2 = auto()       # 1 rendered, 2 black (33% density)
    VERY_SPARSE = auto()    # 1 rendered, 3 black (25% density)


@dataclass(frozen=True)
class DegradationZone:
    """
    Represents a single degradation zone in foveated rendering.
    
    Attributes:
        outer_radius: The outer boundary radius of this zone in pixels.
        pattern: The degradation pattern to apply in this zone.
        gradient_width: Width of the gradient transition at zone edges.
    """
    outer_radius: int
    pattern: DegradationPattern
    gradient_width: int = 10
    
    def __post_init__(self):
        if self.outer_radius <= 0:
            raise ValueError(f"outer_radius must be positive, got {self.outer_radius}")
        if self.gradient_width < 0:
            raise ValueError(f"gradient_width cannot be negative, got {self.gradient_width}")


@dataclass
class RenderConfig:
    """
    Complete configuration for foveated rendering.
    
    Attributes:
        window_width: Width of the render window in pixels.
        window_height: Height of the render window in pixels.
        fovea_radius: Radius of the fully-rendered central region.
        zones: List of degradation zones, ordered from inner to outer.
        background_color: RGB color for non-rendered pixels.
        enable_gradients: Whether to enable smooth gradient transitions.
        gradient_strength: Strength of gradient blending (0.0 to 1.0).
    """
    window_width: int = 1280
    window_height: int = 720
    fovea_radius: int = 50  # 100px diameter = 50px radius
    zones: List[DegradationZone] = field(default_factory=list)
    background_color: Tuple[int, int, int] = (0, 0, 0)
    enable_gradients: bool = True
    gradient_strength: float = 0.8
    
    def __post_init__(self):
        if self.window_width <= 0 or self.window_height <= 0:
            raise ValueError("Window dimensions must be positive")
        if self.fovea_radius <= 0:
            raise ValueError("Fovea radius must be positive")
        if not 0.0 <= self.gradient_strength <= 1.0:
            raise ValueError("Gradient strength must be between 0.0 and 1.0")
        
        # Set default zones if none provided
        if not self.zones:
            self.zones = self._create_default_zones()
    
    def _create_default_zones(self) -> List[DegradationZone]:
        """Create default degradation zones based on fovea radius."""
        return [
            DegradationZone(
                outer_radius=self.fovea_radius,
                pattern=DegradationPattern.FULL,
                gradient_width=0
            ),
            DegradationZone(
                outer_radius=self.fovea_radius + 60,
                pattern=DegradationPattern.SPARSE_1,
                gradient_width=15
            ),
            DegradationZone(
                outer_radius=self.fovea_radius + 120,
                pattern=DegradationPattern.CHECKERBOARD,
                gradient_width=15
            ),
            DegradationZone(
                outer_radius=self.fovea_radius + 200,
                pattern=DegradationPattern.SPARSE_2,
                gradient_width=20
            ),
            DegradationZone(
                outer_radius=max(self.window_width, self.window_height) * 2,
                pattern=DegradationPattern.VERY_SPARSE,
                gradient_width=25
            ),
        ]
    
    def get_zone_for_distance(self, distance: float) -> Tuple[DegradationZone, int]:
        """
        Get the appropriate zone for a given distance from fovea center.
        
        Args:
            distance: Distance in pixels from the fovea center.
            
        Returns:
            Tuple of (zone, zone_index) for the given distance.
        """
        for idx, zone in enumerate(self.zones):
            if distance <= zone.outer_radius:
                return zone, idx
        return self.zones[-1], len(self.zones) - 1


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
