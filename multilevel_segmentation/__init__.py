"""
Multilevel Image Segmentation Library

A Python library for multilevel image segmentation that combines 
clustering and morphological operations.
"""

from .segmentation_model import SegmentationModel
from .visualization import visualize_results, compare_methods
from .evaluation import evaluate_segmentation

__version__ = '0.1.0'