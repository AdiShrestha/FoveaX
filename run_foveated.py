#!/usr/bin/env python3
"""
Entry point script for running the foveated rendering application.

Usage:
    python run_foveated.py [--image PATH] [--width W] [--height H] [--fovea-radius R]
    
Examples:
    python run_foveated.py
    python run_foveated.py --image wallpaper.jpg
    python run_foveated.py --width 1920 --height 1080 --fovea-radius 75
"""

import sys
import logging
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from Rendering.application import main

if __name__ == '__main__':
    main()
