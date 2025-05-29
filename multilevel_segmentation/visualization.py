"""
Visualization Module

This module contains functions for visualizing segmentation results.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
import os


def visualize_results(
    original_image: np.ndarray,
    segmentation_result: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (15, 10)
) -> None:
    """
    Visualize segmentation results showing all four stages.
    
    Parameters
    ----------
    original_image : np.ndarray
        The original input image.
    segmentation_result : Dict[str, np.ndarray]
        Dictionary containing segmentation results.
    save_path : str, optional
        Path to save the visualization. If None, it won't be saved.
    show : bool, optional
        Whether to display the visualization. Default is True.
    figsize : Tuple[int, int], optional
        Figure size. Default is (15, 10).
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot original image
    axes[0, 0].imshow(original_image, cmap='gray' if len(original_image.shape) == 2 else None)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Plot clustered image
    axes[0, 1].imshow(segmentation_result['clustered_image'], cmap='viridis')
    axes[0, 1].set_title('Clustered Image')
    axes[0, 1].axis('off')
    
    # Plot morphological image
    axes[1, 0].imshow(segmentation_result['morphological_image'], cmap='viridis')
    axes[1, 0].set_title('Morphological Image')
    axes[1, 0].axis('off')
    
    # Plot combined result
    axes[1, 1].imshow(segmentation_result['combined_image'], cmap='viridis')
    axes[1, 1].set_title('Combined Result')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def save_all_results(
    original_image: np.ndarray,
    segmentation_result: Dict[str, np.ndarray],
    output_dir: str
) -> None:
    """
    Save all segmentation results as separate images.
    
    Parameters
    ----------
    original_image : np.ndarray
        The original input image.
    segmentation_result : Dict[str, np.ndarray]
        Dictionary containing segmentation results.
    output_dir : str
        Directory to save the results.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each result
    plt.imsave(os.path.join(output_dir, 'original.png'), original_image, 
               cmap='gray' if len(original_image.shape) == 2 else None)
    plt.imsave(os.path.join(output_dir, 'clustered.png'), 
               segmentation_result['clustered_image'], cmap='viridis')
    plt.imsave(os.path.join(output_dir, 'morphological.png'), 
               segmentation_result['morphological_image'], cmap='viridis')
    plt.imsave(os.path.join(output_dir, 'combined.png'), 
               segmentation_result['combined_image'], cmap='viridis')