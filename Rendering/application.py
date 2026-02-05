"""
Main application module for foveated rendering demonstration.

Provides a pygame-based interactive window for testing foveated rendering
with mouse-controlled fovea positioning.
"""

import sys
import logging
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

from .config import RenderConfig, create_default_config
from .foveated_renderer import FoveatedRenderer

logger = logging.getLogger(__name__)


class AppState(Enum):
    """Application state enumeration."""
    INITIALIZING = auto()
    RUNNING = auto()
    PAUSED = auto()
    STOPPED = auto()


@dataclass
class AppStats:
    """Runtime statistics for the application."""
    frame_count: int = 0
    total_render_time: float = 0.0
    last_frame_time: float = 0.0
    fps: float = 0.0
    
    def update(self, frame_time: float) -> None:
        """Update statistics with new frame data."""
        self.frame_count += 1
        self.total_render_time += frame_time
        self.last_frame_time = frame_time
        if self.total_render_time > 0:
            self.fps = self.frame_count / self.total_render_time


class ImageLoader:
    """Handles loading and preparing images for rendering."""
    
    @staticmethod
    def load_image(path: Union[str, Path], 
                   target_size: tuple) -> np.ndarray:
        """
        Load and resize an image to target dimensions.
        
        Args:
            path: Path to the image file.
            target_size: Target (width, height) tuple.
            
        Returns:
            Numpy array of the image in RGB format.
            
        Raises:
            FileNotFoundError: If image file doesn't exist.
            ValueError: If image cannot be loaded.
        """
        if not PYGAME_AVAILABLE:
            raise RuntimeError("pygame is required for image loading")
        
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        
        try:
            surface = pygame.image.load(str(path))
            surface = pygame.transform.scale(surface, target_size)
            
            # Convert to numpy array
            array = pygame.surfarray.array3d(surface)
            # Pygame uses (width, height, channels), numpy uses (height, width, channels)
            array = np.transpose(array, (1, 0, 2))
            
            return array
            
        except pygame.error as e:
            raise ValueError(f"Failed to load image: {e}")
    
    @staticmethod
    def create_gradient_image(width: int, height: int) -> np.ndarray:
        """
        Create a colorful gradient image for testing.
        
        Args:
            width: Image width.
            height: Image height.
            
        Returns:
            Numpy array with gradient image.
        """
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create a nice gradient pattern
        for y in range(height):
            for x in range(width):
                # Colorful diagonal gradient
                r = int(255 * (x / width))
                g = int(255 * (y / height))
                b = int(255 * ((x + y) / (width + height)))
                image[y, x] = [r, g, b]
        
        return image
    
    @staticmethod
    def create_test_pattern(width: int, height: int) -> np.ndarray:
        """
        Create a test pattern with various elements for visualizing degradation.
        
        Args:
            width: Image width.
            height: Image height.
            
        Returns:
            Numpy array with test pattern.
        """
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Background gradient
        y_coords, x_coords = np.ogrid[:height, :width]
        image[:, :, 0] = (x_coords * 255 // width).astype(np.uint8)
        image[:, :, 1] = (y_coords * 255 // height).astype(np.uint8)
        image[:, :, 2] = 128
        
        # Add grid lines
        grid_spacing = 50
        image[::grid_spacing, :] = [255, 255, 255]
        image[:, ::grid_spacing] = [255, 255, 255]
        
        # Add circles at regular intervals
        center_y, center_x = height // 2, width // 2
        for radius in range(50, max(width, height), 100):
            # Draw circle outline
            theta = np.linspace(0, 2 * np.pi, 360)
            for t in theta:
                cx = int(center_x + radius * np.cos(t))
                cy = int(center_y + radius * np.sin(t))
                if 0 <= cx < width and 0 <= cy < height:
                    image[cy, cx] = [255, 255, 0]
        
        # Add some text-like patterns (horizontal lines of varying thickness)
        for y_offset in range(100, height - 100, 80):
            thickness = (y_offset // 80) % 3 + 1
            for t in range(thickness):
                if y_offset + t < height:
                    image[y_offset + t, 100:width-100] = [255, 255, 255]
        
        return image


class FoveatedRenderingApp:
    """
    Main application class for interactive foveated rendering.
    
    Provides a pygame window with mouse-controlled fovea positioning
    and real-time rendering preview.
    """
    
    def __init__(self, 
                 config: Optional[RenderConfig] = None,
                 image_path: Optional[Union[str, Path]] = None):
        """
        Initialize the foveated rendering application.
        
        Args:
            config: Rendering configuration (uses defaults if None).
            image_path: Optional path to an image file to display.
        """
        if not PYGAME_AVAILABLE:
            raise RuntimeError(
                "pygame is required for the application. "
                "Install with: pip install pygame"
            )
        
        self.config = config or create_default_config()
        self.image_path = image_path
        self.state = AppState.INITIALIZING
        self.stats = AppStats()
        
        # Initialize components
        self._screen: Optional[pygame.Surface] = None
        self._clock: Optional[pygame.time.Clock] = None
        self._source_image: Optional[np.ndarray] = None
        self._renderer: Optional[FoveatedRenderer] = None
        self._rendered_surface: Optional[pygame.Surface] = None
        
        # Interaction state
        self._fovea_locked = False
        self._show_debug = False
        self._continuous_update = True
        
        logger.info("FoveatedRenderingApp initialized")
    
    def _init_pygame(self) -> None:
        """Initialize pygame and create the window."""
        pygame.init()
        pygame.display.set_caption("Foveated Rendering - Click to set focus point")
        
        self._screen = pygame.display.set_mode(
            (self.config.window_width, self.config.window_height)
        )
        self._clock = pygame.time.Clock()
        
        logger.info(
            f"Pygame initialized: {self.config.window_width}x{self.config.window_height}"
        )
    
    def _load_source_image(self) -> None:
        """Load or generate the source image."""
        if self.image_path:
            try:
                self._source_image = ImageLoader.load_image(
                    self.image_path,
                    (self.config.window_width, self.config.window_height)
                )
                logger.info(f"Loaded image: {self.image_path}")
            except (FileNotFoundError, ValueError) as e:
                logger.warning(f"Failed to load image: {e}. Using test pattern.")
                self._source_image = ImageLoader.create_test_pattern(
                    self.config.window_width,
                    self.config.window_height
                )
        else:
            # Create test pattern
            self._source_image = ImageLoader.create_test_pattern(
                self.config.window_width,
                self.config.window_height
            )
            logger.info("Using generated test pattern")
    
    def _init_renderer(self) -> None:
        """Initialize the foveated renderer."""
        self._renderer = FoveatedRenderer(self.config)
        logger.info("Foveated renderer initialized")
    
    def _numpy_to_surface(self, array: np.ndarray) -> pygame.Surface:
        """
        Convert numpy array to pygame surface.
        
        Args:
            array: Numpy array in (height, width, channels) format.
            
        Returns:
            Pygame surface.
        """
        # Transpose from (H, W, C) to (W, H, C) for pygame
        transposed = np.transpose(array, (1, 0, 2))
        return pygame.surfarray.make_surface(transposed)
    
    def _render_frame(self) -> None:
        """Render a single frame."""
        if self._renderer is None or self._source_image is None:
            return
        
        import time
        start_time = time.perf_counter()
        
        # Apply foveated rendering
        rendered = self._renderer.render_optimized(self._source_image)
        
        # Convert to pygame surface
        self._rendered_surface = self._numpy_to_surface(rendered)
        
        # Update stats
        frame_time = time.perf_counter() - start_time
        self.stats.update(frame_time)
    
    def _draw_debug_overlay(self) -> None:
        """Draw debug information overlay."""
        if not self._show_debug or self._screen is None:
            return
        
        font = pygame.font.Font(None, 24)
        
        # Prepare debug text
        lines = [
            f"FPS: {self.stats.fps:.1f}",
            f"Frame time: {self.stats.last_frame_time * 1000:.2f}ms",
            f"Fovea: {self._renderer.fovea.x}, {self._renderer.fovea.y}" 
                if self._renderer and self._renderer.fovea else "Fovea: None",
            f"Locked: {self._fovea_locked}",
            "",
            "Controls:",
            "Click - Set fovea position",
            "Space - Toggle fovea lock",
            "D - Toggle debug overlay",
            "R - Reset fovea",
            "ESC - Quit"
        ]
        
        # Draw semi-transparent background
        overlay_height = len(lines) * 22 + 10
        overlay_surface = pygame.Surface((220, overlay_height))
        overlay_surface.set_alpha(180)
        overlay_surface.fill((0, 0, 0))
        self._screen.blit(overlay_surface, (10, 10))
        
        # Draw text
        y_offset = 15
        for line in lines:
            text_surface = font.render(line, True, (255, 255, 255))
            self._screen.blit(text_surface, (15, y_offset))
            y_offset += 22
    
    def _draw_fovea_indicator(self) -> None:
        """Draw a subtle indicator at the fovea position."""
        if self._renderer is None or self._renderer.fovea is None:
            return
        if self._screen is None:
            return
        
        fovea = self._renderer.fovea
        
        # Draw crosshair
        color = (255, 255, 255) if not self._fovea_locked else (0, 255, 0)
        size = 10
        
        pygame.draw.line(
            self._screen, color,
            (fovea.x - size, fovea.y),
            (fovea.x + size, fovea.y),
            1
        )
        pygame.draw.line(
            self._screen, color,
            (fovea.x, fovea.y - size),
            (fovea.x, fovea.y + size),
            1
        )
        
        # Draw zone circles (subtle)
        if self._show_debug:
            for zone in self.config.zones[:4]:  # Only show first 4 zones
                pygame.draw.circle(
                    self._screen,
                    (100, 100, 100),
                    (fovea.x, fovea.y),
                    zone.outer_radius,
                    1
                )
    
    def _handle_events(self) -> bool:
        """
        Handle pygame events.
        
        Returns:
            False if application should quit, True otherwise.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE:
                    self._fovea_locked = not self._fovea_locked
                    logger.debug(f"Fovea lock: {self._fovea_locked}")
                elif event.key == pygame.K_d:
                    self._show_debug = not self._show_debug
                elif event.key == pygame.K_r:
                    if self._renderer:
                        self._renderer.clear_fovea()
                        self._fovea_locked = False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    if self._renderer:
                        self._renderer.set_fovea(event.pos[0], event.pos[1])
        
        # Handle continuous mouse tracking (if not locked)
        if not self._fovea_locked and self._continuous_update:
            if pygame.mouse.get_focused():
                mouse_pos = pygame.mouse.get_pos()
                if self._renderer:
                    self._renderer.set_fovea(mouse_pos[0], mouse_pos[1])
        
        return True
    
    def run(self) -> None:
        """
        Run the main application loop.
        
        This method blocks until the application is closed.
        """
        try:
            # Initialize
            self._init_pygame()
            self._load_source_image()
            self._init_renderer()
            
            self.state = AppState.RUNNING
            logger.info("Application started")
            
            # Main loop
            running = True
            while running:
                # Handle events
                running = self._handle_events()
                
                # Render
                self._render_frame()
                
                # Display
                if self._rendered_surface and self._screen:
                    self._screen.blit(self._rendered_surface, (0, 0))
                    self._draw_fovea_indicator()
                    self._draw_debug_overlay()
                    pygame.display.flip()
                
                # Cap frame rate
                self._clock.tick(60)
            
        except Exception as e:
            logger.error(f"Application error: {e}", exc_info=True)
            raise
        
        finally:
            self.state = AppState.STOPPED
            pygame.quit()
            logger.info("Application stopped")
    
    def run_single_frame(self, fovea_x: int, fovea_y: int) -> np.ndarray:
        """
        Render a single frame without running the full application loop.
        
        Useful for testing or integration with other systems.
        
        Args:
            fovea_x: X coordinate of the fovea.
            fovea_y: Y coordinate of the fovea.
            
        Returns:
            Rendered image as numpy array.
        """
        if self._source_image is None:
            self._load_source_image()
        
        if self._renderer is None:
            self._init_renderer()
        
        self._renderer.set_fovea(fovea_x, fovea_y)
        return self._renderer.render_optimized(self._source_image)


def main():
    """Main entry point for the application."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(
        description='Foveated Rendering Demonstration'
    )
    parser.add_argument(
        '--image', '-i',
        type=str,
        help='Path to an image file to display'
    )
    parser.add_argument(
        '--width', '-W',
        type=int,
        default=1280,
        help='Window width (default: 1280)'
    )
    parser.add_argument(
        '--height', '-H',
        type=int,
        default=720,
        help='Window height (default: 720)'
    )
    parser.add_argument(
        '--fovea-radius', '-r',
        type=int,
        default=50,
        help='Fovea radius in pixels (default: 50)'
    )
    parser.add_argument(
        '--no-gradients',
        action='store_true',
        help='Disable gradient transitions between zones'
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = RenderConfig(
        window_width=args.width,
        window_height=args.height,
        fovea_radius=args.fovea_radius,
        enable_gradients=not args.no_gradients
    )
    
    # Run application
    app = FoveatedRenderingApp(config=config, image_path=args.image)
    app.run()


if __name__ == '__main__':
    main()
