"""
Foveated Rendering Package

A modular implementation of progressive circular foveated rendering
with seamless transitions and continuous degradation.
"""

from .config import RenderConfig
from .foveated_renderer import FoveatedRenderer
from .application import FoveatedRenderingApp

__all__ = [
    'RenderConfig',
    'FoveatedRenderer',
    'FoveatedRenderingApp'
]

__version__ = '1.1.0'
