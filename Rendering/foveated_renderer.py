"""
Core foveated rendering engine.

Implements truly continuous progressive foveated rendering with seamless
transitions. Pixel density decreases smoothly with distance - no discrete
levels or visible boundaries.
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass
import logging

from .config import RenderConfig

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


class FoveatedRenderer:
    """
    Foveated rendering engine with truly continuous seamless degradation.
    
    Uses distance-based density calculation with ordered dithering to create
    perfectly smooth transitions. Degradation never stops - continues all
    the way to screen corners.
    """
    
    def __init__(self, config: RenderConfig):
        """
        Initialize the foveated renderer.
        
        Args:
            config: Rendering configuration.
        """
        self.config = config
        self.fovea: Optional[FoveaState] = None
        
        # Pre-compute coordinate grids
        self._x_grid, self._y_grid = np.meshgrid(
            np.arange(config.window_width),
            np.arange(config.window_height)
        )
        
        # Create ordered dithering matrix (8x8 Bayer matrix for smooth transitions)
        self._dither_matrix = self._create_dither_matrix(8)
        
        # Tile the dither matrix to cover the full screen
        self._dither_map = self._create_dither_map()
        
        # Distance cache
        self._distance_cache: Optional[np.ndarray] = None
        
        logger.info(f"FoveatedRenderer initialized - fovea radius: {config.fovea_radius}px")
    
    def _create_dither_matrix(self, size: int) -> np.ndarray:
        """
        Create a Bayer ordered dithering matrix for smooth transitions.
        
        Args:
            size: Size of the matrix (must be power of 2).
            
        Returns:
            Normalized dither matrix (values 0 to 1).
        """
        if size == 2:
            return np.array([[0, 2], [3, 1]]) / 4.0
        
        smaller = self._create_dither_matrix(size // 2)
        
        # Recursive construction of Bayer matrix
        matrix = np.zeros((size, size))
        matrix[:size//2, :size//2] = 4 * smaller
        matrix[:size//2, size//2:] = 4 * smaller + 2
        matrix[size//2:, :size//2] = 4 * smaller + 3
        matrix[size//2:, size//2:] = 4 * smaller + 1
        
        return matrix / (size * size)
    
    def _create_dither_map(self) -> np.ndarray:
        """Create a full-screen dither threshold map."""
        h, w = self.config.window_height, self.config.window_width
        dither_size = self._dither_matrix.shape[0]
        
        # Tile the dither matrix across the screen
        tiles_x = (w + dither_size - 1) // dither_size
        tiles_y = (h + dither_size - 1) // dither_size
        
        tiled = np.tile(self._dither_matrix, (tiles_y, tiles_x))
        return tiled[:h, :w]
    
    def set_fovea(self, x: int, y: int) -> None:
        """
        Set the fovea (gaze) position.
        
        Args:
            x: X coordinate of the fovea center.
            y: Y coordinate of the fovea center.
        """
        x = max(0, min(x, self.config.window_width - 1))
        y = max(0, min(y, self.config.window_height - 1))
        
        if self.fovea is None:
            self.fovea = FoveaState(x, y)
        else:
            self.fovea.update(x, y)
        
        self._distance_cache = None
        logger.debug(f"Fovea set to ({x}, {y})")
    
    def clear_fovea(self) -> None:
        """Clear the fovea position."""
        self.fovea = None
        self._distance_cache = None
    
    def _compute_distance_map(self) -> np.ndarray:
        """Compute distance map from current fovea position."""
        if self._distance_cache is not None:
            return self._distance_cache
        
        if self.fovea is None:
            return np.full(
                (self.config.window_height, self.config.window_width),
                float('inf')
            )
        
        dx = self._x_grid - self.fovea.x
        dy = self._y_grid - self.fovea.y
        self._distance_cache = np.sqrt(dx * dx + dy * dy)
        
        return self._distance_cache
    
    def _compute_density_map(self, distance_map: np.ndarray) -> np.ndarray:
        """
        Compute pixel density based on distance.
        
        Uses a smooth function that:
        - Returns 1.0 (100%) within fovea radius
        - Gradually decreases beyond fovea
        - Never reaches 0 but gets very sparse at far distances
        
        The degradation starts very subtle (like 10:1 ratio) and progressively
        increases (9:1, 8:1, ... 2:1, 1:1, 1:2, 1:3, etc.)
        
        Args:
            distance_map: Distance from fovea for each pixel.
            
        Returns:
            Density values (0 to 1) for each pixel.
        """
        fovea_r = self.config.fovea_radius
        
        # Full density within fovea
        density = np.ones_like(distance_map)
        
        # Beyond fovea, density decreases with distance
        beyond_fovea = distance_map > fovea_r
        
        if np.any(beyond_fovea):
            # Calculate excess distance (how far beyond fovea)
            excess = distance_map[beyond_fovea] - fovea_r
            
            # Use inverse relationship for smooth continuous falloff
            # density = fovea_r / (fovea_r + excess * rate)
            # This gives: at excess=0 -> density=1, as excess->inf -> density->0
            # The rate controls how fast it falls off
            rate = self.config.degradation_rate
            
            # Smooth continuous density function
            # Starts subtle and continuously decreases
            density[beyond_fovea] = fovea_r / (fovea_r + excess * rate)
        
        return density
    
    def _create_continuous_mask(self, distance_map: np.ndarray) -> np.ndarray:
        """
        Create a continuous rendering mask using ordered dithering.
        
        The mask smoothly transitions based on distance - no discrete levels
        or visible boundaries.
        
        Args:
            distance_map: Pre-computed distance from fovea.
            
        Returns:
            Boolean mask where True = render pixel.
        """
        # Get density (0 to 1) for each pixel
        density = self._compute_density_map(distance_map)
        
        # Use ordered dithering: render pixel if density > dither threshold
        # This creates smooth transitions without visible banding
        mask = density > self._dither_map
        
        return mask
    
    def _apply_edge_smoothing(self, output: np.ndarray, 
                              source: np.ndarray,
                              mask: np.ndarray,
                              distance_map: np.ndarray) -> np.ndarray:
        """
        Apply subtle opacity fade at very far distances.
        
        Args:
            output: Current output image.
            source: Source image.
            mask: Render mask.
            distance_map: Distance from fovea.
            
        Returns:
            Output with smoothing applied.
        """
        if not self.config.enable_gradients:
            return output
        
        # Calculate max possible distance (corner to corner)
        max_dist = np.sqrt(self.config.window_width**2 + self.config.window_height**2)
        
        # Very subtle fade at extreme distances
        fade_start = max_dist * 0.5
        
        fade_mask = mask & (distance_map > fade_start)
        if np.any(fade_mask):
            # Subtle opacity reduction at far distances
            fade_progress = (distance_map[fade_mask] - fade_start) / (max_dist - fade_start)
            opacity = 1.0 - (fade_progress * 0.3)  # Fade to 70% at max distance
            opacity = np.clip(opacity, 0.7, 1.0)
            
            for c in range(min(3, output.shape[2])):
                blended = (
                    source[:, :, c][fade_mask].astype(float) * opacity +
                    self.config.background_color[c] * (1 - opacity)
                ).astype(np.uint8)
                output[:, :, c][fade_mask] = blended
        
        return output
    
    def render(self, source_image: np.ndarray) -> np.ndarray:
        """
        Apply foveated rendering to a source image.
        
        Args:
            source_image: Input image as numpy array (H, W, 3) or (H, W, 4).
            
        Returns:
            Rendered image with continuous foveated degradation.
            
        Raises:
            ValueError: If image dimensions don't match config.
        """
        if source_image.shape[0] != self.config.window_height or \
           source_image.shape[1] != self.config.window_width:
            raise ValueError(
                f"Image dimensions {source_image.shape[:2]} don't match "
                f"config ({self.config.window_height}, {self.config.window_width})"
            )
        
        has_alpha = len(source_image.shape) > 2 and source_image.shape[2] == 4
        
        # Initialize output with background
        output = np.zeros_like(source_image)
        if len(source_image.shape) > 2:
            output[:, :, :3] = self.config.background_color
            if has_alpha:
                output[:, :, 3] = 255
        
        # No fovea - render with maximum distance assumption
        if self.fovea is None or not self.fovea.active:
            # Assume fovea is off-screen, use very sparse rendering
            max_dist = np.sqrt(self.config.window_width**2 + self.config.window_height**2)
            fake_distance = np.full_like(self._dither_map, max_dist)
            render_mask = self._create_continuous_mask(fake_distance)
            output[render_mask] = source_image[render_mask]
            return output
        
        # Compute distance and create continuous mask
        distance_map = self._compute_distance_map()
        render_mask = self._create_continuous_mask(distance_map)
        
        # Apply mask
        output[render_mask] = source_image[render_mask]
        
        # Apply subtle edge smoothing
        output = self._apply_edge_smoothing(output, source_image, render_mask, distance_map)
        
        return output
    
    def render_optimized(self, source_image: np.ndarray) -> np.ndarray:
        """
        Optimized rendering - same as render() for API compatibility.
        
        Args:
            source_image: Input image.
            
        Returns:
            Rendered image.
        """
        return self.render(source_image)
    
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
            'fovea_radius': self.config.fovea_radius,
        }
        
        if self.fovea:
            distance_map = self._compute_distance_map()
            render_mask = self._create_continuous_mask(distance_map)
            rendered_pixels = np.sum(render_mask)
            
            stats['rendered_pixels'] = int(rendered_pixels)
            stats['render_percentage'] = round(100 * rendered_pixels / total_pixels, 2)
            stats['pixels_saved'] = total_pixels - int(rendered_pixels)
            stats['efficiency'] = round(100 * (1 - rendered_pixels / total_pixels), 2)
        
        return stats
