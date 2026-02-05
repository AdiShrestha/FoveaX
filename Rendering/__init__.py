"""
Foveated Rendering Package

A modular implementation of circular foveated rendering with configurable
degradation zones and smooth gradient transitions.
"""

from .config import RenderConfig, DegradationZone
from .foveated_renderer import FoveatedRenderer
from .application import FoveatedRenderingApp

__all__ = [
    'RenderConfig',
    'DegradationZone', 
    'FoveatedRenderer',
    'FoveatedRenderingApp'
]

__version__ = '1.0.0'
