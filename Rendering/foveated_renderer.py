"""
Core foveated rendering engine.

Implements the pixel-level rendering logic with configurable degradation patterns
and smooth gradient transitions between zones.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
import logging

from .config import RenderConfig, DegradationPattern, DegradationZone

logger = logging.getLogger(__name__)


@dataclass
class FoveaState:
    """Represents the current state of the fovea (gaze point)."""
    x: int
    y: int
    active: bool = True
    
    def update(self, x: int, y: int) -> None:
        """Update fovea position."""
        self.x = x
        self.y = y
        self.active = True


class PatternGenerator:
    """
    Generates pixel mask patterns for different degradation levels.
    
    Uses numpy for efficient array operations.
    """
    
    @staticmethod
    def should_render_pixel(x: int, y: int, pattern: DegradationPattern) -> bool:
        """
        Determine if a pixel should be rendered based on pattern.
        
        Args:
            x: Pixel x coordinate.
            y: Pixel y coordinate.
            pattern: The degradation pattern to apply.
            
        Returns:
            True if the pixel should be rendered, False for black.
        """
        if pattern == DegradationPattern.FULL:
            return True
        
        elif pattern == DegradationPattern.SPARSE_1:
            # 2 rendered, 1 black (repeating pattern of 3)
            # Pattern: RR.RR.RR. (66% density)
            return (x + y) % 3 != 0
        
        elif pattern == DegradationPattern.CHECKERBOARD:
            # Standard checkerboard: 1 rendered, 1 black (50% density)
            return (x + y) % 2 == 0
        
        elif pattern == DegradationPattern.SPARSE_2:
            # 1 rendered, 2 black (repeating pattern of 3)
            # Pattern: R..R..R.. (33% density)
            return (x + y) % 3 == 0
        
        elif pattern == DegradationPattern.VERY_SPARSE:
            # 1 rendered, 3 black (25% density)
            return (x % 2 == 0) and (y % 2 == 0)
        
        return True
    
    @staticmethod
    def create_pattern_mask(width: int, height: int, 
                           pattern: DegradationPattern) -> np.ndarray:
        """
        Create a full mask array for a given pattern.
        
        Args:
            width: Width of the mask.
            height: Height of the mask.
            pattern: The degradation pattern.
            
        Returns:
            Boolean numpy array where True = render pixel.
        """
        y_coords, x_coords = np.ogrid[:height, :width]
        
        if pattern == DegradationPattern.FULL:
            return np.ones((height, width), dtype=bool)
        
        elif pattern == DegradationPattern.SPARSE_1:
            return (x_coords + y_coords) % 3 != 0
        
        elif pattern == DegradationPattern.CHECKERBOARD:
            return (x_coords + y_coords) % 2 == 0
        
        elif pattern == DegradationPattern.SPARSE_2:
            return (x_coords + y_coords) % 3 == 0
        
        elif pattern == DegradationPattern.VERY_SPARSE:
            return (x_coords % 2 == 0) & (y_coords % 2 == 0)
        
        return np.ones((height, width), dtype=bool)


class FoveatedRenderer:
    """
    Main foveated rendering engine.
    
    Applies circular foveated rendering with configurable degradation zones
    and smooth gradient transitions.
    """
    
    def __init__(self, config: RenderConfig):
        """
        Initialize the foveated renderer.
        
        Args:
            config: Rendering configuration.
        """
        self.config = config
        self.fovea: Optional[FoveaState] = None
        self._pattern_generator = PatternGenerator()
        
        # Pre-compute pattern masks for efficiency
        self._pattern_masks = self._precompute_pattern_masks()
        
        # Pre-compute distance lookup table
        self._distance_cache: Optional[np.ndarray] = None
        
        logger.info(f"FoveatedRenderer initialized with {len(config.zones)} zones")
    
    def _precompute_pattern_masks(self) -> dict:
        """Pre-compute pattern masks for all degradation patterns."""
        masks = {}
        for pattern in DegradationPattern:
            masks[pattern] = PatternGenerator.create_pattern_mask(
                self.config.window_width,
                self.config.window_height,
                pattern
            )
        return masks
    
    def set_fovea(self, x: int, y: int) -> None:
        """
        Set the fovea (gaze) position.
        
        Args:
            x: X coordinate of the fovea center.
            y: Y coordinate of the fovea center.
        """
        # Clamp to valid range
        x = max(0, min(x, self.config.window_width - 1))
        y = max(0, min(y, self.config.window_height - 1))
        
        if self.fovea is None:
            self.fovea = FoveaState(x, y)
        else:
            self.fovea.update(x, y)
        
        # Invalidate distance cache
        self._distance_cache = None
        
        logger.debug(f"Fovea set to ({x}, {y})")
    
    def clear_fovea(self) -> None:
        """Clear the fovea position (no focal point)."""
        self.fovea = None
        self._distance_cache = None
    
    def _compute_distance_map(self) -> np.ndarray:
        """
        Compute distance map from current fovea position.
        
        Returns:
            2D array of distances from each pixel to the fovea center.
        """
        if self._distance_cache is not None:
            return self._distance_cache
        
        if self.fovea is None:
            # No fovea - return max distance everywhere
            return np.full(
                (self.config.window_height, self.config.window_width),
                float('inf')
            )
        
        y_coords, x_coords = np.ogrid[:self.config.window_height, 
                                       :self.config.window_width]
        
        # Euclidean distance from fovea center
        dx = x_coords - self.fovea.x
        dy = y_coords - self.fovea.y
        self._distance_cache = np.sqrt(dx * dx + dy * dy)
        
        return self._distance_cache
    
    def _compute_zone_mask(self, distance_map: np.ndarray, 
                          zone: DegradationZone,
                          inner_radius: float) -> np.ndarray:
        """
        Compute the mask for a specific zone.
        
        Args:
            distance_map: Pre-computed distance map.
            zone: The zone to compute mask for.
            inner_radius: Inner radius of this zone.
            
        Returns:
            Boolean mask where True = pixel is in this zone.
        """
        return (distance_map > inner_radius) & (distance_map <= zone.outer_radius)
    
    def _compute_gradient_factor(self, distance_map: np.ndarray,
                                 zone: DegradationZone,
                                 inner_radius: float) -> np.ndarray:
        """
        Compute gradient blending factors for zone transitions.
        
        Args:
            distance_map: Pre-computed distance map.
            zone: The zone configuration.
            inner_radius: Inner radius of this zone.
            
        Returns:
            Float array of gradient factors (0.0 to 1.0).
        """
        if not self.config.enable_gradients or zone.gradient_width == 0:
            return np.ones_like(distance_map)
        
        gradient = np.ones_like(distance_map)
        
        # Gradient at inner edge (fade in)
        if inner_radius > 0:
            inner_gradient_end = inner_radius + zone.gradient_width
            inner_mask = (distance_map > inner_radius) & (distance_map < inner_gradient_end)
            if np.any(inner_mask):
                gradient[inner_mask] = (
                    (distance_map[inner_mask] - inner_radius) / zone.gradient_width
                )
        
        # Gradient at outer edge (fade out)
        outer_gradient_start = zone.outer_radius - zone.gradient_width
        outer_mask = (distance_map > outer_gradient_start) & (distance_map <= zone.outer_radius)
        if np.any(outer_mask):
            gradient[outer_mask] = (
                (zone.outer_radius - distance_map[outer_mask]) / zone.gradient_width
            )
        
        return np.clip(gradient * self.config.gradient_strength + 
                      (1 - self.config.gradient_strength), 0, 1)
    
    def render(self, source_image: np.ndarray) -> np.ndarray:
        """
        Apply foveated rendering to a source image.
        
        Args:
            source_image: Input image as numpy array (H, W, 3) or (H, W, 4).
            
        Returns:
            Rendered image with foveated degradation applied.
            
        Raises:
            ValueError: If source image dimensions don't match config.
        """
        if source_image.shape[0] != self.config.window_height or \
           source_image.shape[1] != self.config.window_width:
            raise ValueError(
                f"Image dimensions {source_image.shape[:2]} don't match "
                f"config ({self.config.window_height}, {self.config.window_width})"
            )
        
        # Handle both RGB and RGBA images
        has_alpha = source_image.shape[2] == 4 if len(source_image.shape) > 2 else False
        
        # Create output image initialized with background color
        output = np.zeros_like(source_image)
        if len(source_image.shape) > 2:
            output[:, :, :3] = self.config.background_color
            if has_alpha:
                output[:, :, 3] = 255
        
        # If no fovea set, return degraded image based on outermost zone
        if self.fovea is None or not self.fovea.active:
            outermost_pattern = self.config.zones[-1].pattern
            mask = self._pattern_masks[outermost_pattern]
            output[mask] = source_image[mask]
            return output
        
        # Compute distance map
        distance_map = self._compute_distance_map()
        
        # Process each zone
        inner_radius = 0.0
        for zone in self.config.zones:
            # Get zone mask
            zone_mask = self._compute_zone_mask(distance_map, zone, inner_radius)
            
            if not np.any(zone_mask):
                inner_radius = zone.outer_radius
                continue
            
            # Get pattern mask for this zone
            pattern_mask = self._pattern_masks[zone.pattern]
            
            # Combine zone and pattern masks
            combined_mask = zone_mask & pattern_mask
            
            # Apply gradient if enabled
            if self.config.enable_gradients and zone.gradient_width > 0:
                gradient_factor = self._compute_gradient_factor(
                    distance_map, zone, inner_radius
                )
                
                # Apply with gradient blending
                for c in range(min(3, source_image.shape[2])):
                    blended = (
                        source_image[:, :, c].astype(float) * gradient_factor +
                        self.config.background_color[c] * (1 - gradient_factor)
                    ).astype(np.uint8)
                    output[:, :, c] = np.where(combined_mask, blended, output[:, :, c])
            else:
                # Direct copy without gradient
                output[combined_mask] = source_image[combined_mask]
            
            inner_radius = zone.outer_radius
        
        return output
    
    def render_optimized(self, source_image: np.ndarray) -> np.ndarray:
        """
        Optimized rendering path using vectorized operations.
        
        This method is faster for large images but uses more memory.
        
        Args:
            source_image: Input image as numpy array.
            
        Returns:
            Rendered image with foveated degradation applied.
        """
        if self.fovea is None or not self.fovea.active:
            return self.render(source_image)
        
        height, width = source_image.shape[:2]
        
        # Compute distance map
        distance_map = self._compute_distance_map()
        
        # Create output initialized with background
        output = np.zeros_like(source_image)
        output[:, :, :3] = self.config.background_color
        if source_image.shape[2] == 4:
            output[:, :, 3] = 255
        
        # Create combined render mask
        render_mask = np.zeros((height, width), dtype=bool)
        gradient_map = np.ones((height, width), dtype=float)
        
        inner_radius = 0.0
        for zone in self.config.zones:
            zone_mask = self._compute_zone_mask(distance_map, zone, inner_radius)
            pattern_mask = self._pattern_masks[zone.pattern]
            
            combined = zone_mask & pattern_mask
            render_mask |= combined
            
            if self.config.enable_gradients and zone.gradient_width > 0:
                zone_gradient = self._compute_gradient_factor(
                    distance_map, zone, inner_radius
                )
                gradient_map = np.where(zone_mask, zone_gradient, gradient_map)
            
            inner_radius = zone.outer_radius
        
        # Apply rendering with gradients
        for c in range(min(3, source_image.shape[2])):
            blended = (
                source_image[:, :, c].astype(float) * gradient_map +
                self.config.background_color[c] * (1 - gradient_map)
            ).astype(np.uint8)
            output[:, :, c] = np.where(render_mask, blended, output[:, :, c])
        
        return output
    
    def get_rendering_stats(self) -> dict:
        """
        Get statistics about the current rendering state.
        
        Returns:
            Dictionary with rendering statistics.
        """
        total_pixels = self.config.window_width * self.config.window_height
        
        stats = {
            'total_pixels': total_pixels,
            'fovea_active': self.fovea is not None and self.fovea.active,
            'fovea_position': (self.fovea.x, self.fovea.y) if self.fovea else None,
            'num_zones': len(self.config.zones),
            'zones': []
        }
        
        if self.fovea:
            distance_map = self._compute_distance_map()
            inner_radius = 0.0
            
            for i, zone in enumerate(self.config.zones):
                zone_mask = self._compute_zone_mask(distance_map, zone, inner_radius)
                pattern_mask = self._pattern_masks[zone.pattern]
                rendered_pixels = np.sum(zone_mask & pattern_mask)
                
                stats['zones'].append({
                    'index': i,
                    'pattern': zone.pattern.name,
                    'outer_radius': zone.outer_radius,
                    'pixels_in_zone': int(np.sum(zone_mask)),
                    'rendered_pixels': int(rendered_pixels)
                })
                
                inner_radius = zone.outer_radius
        
        return stats
