"""
Utility Functions

This module contains utility functions for image processing and segmentation.
"""

import numpy as np
from typing import Optional, Tuple


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image values to the range [0, 1].
    
    Parameters
    ----------
    image : np.ndarray
        The input image.
        
    Returns
    -------
    np.ndarray
        The normalized image.
    """
    img_min = np.min(image)
    img_max = np.max(image)
    
    if img_max == img_min:
        return np.zeros_like(image, dtype=float)
        
    return (image - img_min) / (img_max - img_min)


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image to grayscale.
    
    Parameters
    ----------
    image : np.ndarray
        The input RGB image.
        
    Returns
    -------
    np.ndarray
        The grayscale image.
    """
    if len(image.shape) == 2:
        # Already grayscale
        return image
    
    if image.shape[2] == 4:  # With alpha channel
        # Use the standard RGB to grayscale conversion formula
        # Luminance (Y) = 0.299*R + 0.587*G + 0.114*B
        return np.dot(image[..., :3], [0.299, 0.587, 0.114])
    else:
        # Standard RGB to grayscale
        return np.dot(image[..., :3], [0.299, 0.587, 0.114])


def threshold_image(
    image: np.ndarray,
    threshold: Optional[float] = None,
    method: str = 'otsu'
) -> Tuple[np.ndarray, float]:
    """
    Threshold an image using either a fixed threshold or Otsu's method.
    
    Parameters
    ----------
    image : np.ndarray
        The input image (grayscale).
    threshold : float, optional
        The threshold value. If None, Otsu's method will be used.
    method : str, optional
        The thresholding method. Options are 'fixed' or 'otsu'. Default is 'otsu'.
        
    Returns
    -------
    Tuple[np.ndarray, float]
        A tuple containing the thresholded image and the threshold value.
    """
    if method == 'otsu' and threshold is None:
        threshold = _otsu_threshold(image)
    elif threshold is None:
        # Default to simple mean if neither Otsu nor threshold is specified
        threshold = np.mean(image)
    
    binary = image > threshold
    return binary, threshold


def _otsu_threshold(image: np.ndarray) -> float:
    """
    Calculate optimal threshold using Otsu's method.
    
    Parameters
    ----------
    image : np.ndarray
        The input image (grayscale, normalized to [0, 1]).
        
    Returns
    -------
    float
        The optimal threshold value.
    """
    # Create histogram
    pixel_counts = np.bincount(np.round(image * 255).astype(np.int64).flatten())
    
    # Ensure histogram has 256 bins
    if len(pixel_counts) < 256:
        pixel_counts = np.pad(pixel_counts, (0, 256 - len(pixel_counts)), 'constant')
    elif len(pixel_counts) > 256:
        pixel_counts = pixel_counts[:256]
    
    # Normalize histogram
    pixel_probs = pixel_counts / np.sum(pixel_counts)
    
    # Compute cumulative sums and means
    cumsum = np.cumsum(pixel_probs)
    cumsum_mean = np.cumsum(pixel_probs * np.arange(256))
    global_mean = cumsum_mean[-1]
    
    # Calculate between-class variance
    between_var = (global_mean * cumsum - cumsum_mean) ** 2 / (cumsum * (1 - cumsum) + 1e-10)
    
    # Find threshold that maximizes between-class variance
    max_idx = np.argmax(between_var[1:-1]) + 1  # Avoid edges
    threshold = max_idx / 255.0
    
    return threshold


def calculate_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate Mean Squared Error between two images.
    
    Parameters
    ----------
    img1 : np.ndarray
        First image.
    img2 : np.ndarray
        Second image.
        
    Returns
    -------
    float
        Mean Squared Error.
    """
    return np.mean((img1 - img2) ** 2)


def calculate_psnr(img1: np.ndarray, img2: np.ndarray, max_value: float = 1.0) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio between two images.
    
    Parameters
    ----------
    img1 : np.ndarray
        First image.
    img2 : np.ndarray
        Second image.
    max_value : float, optional
        Maximum possible pixel value. Default is 1.0.
        
    Returns
    -------
    float
        PSNR value in decibels.
    """
    mse = calculate_mse(img1, img2)
    if mse == 0:
        return float('inf')
    
    return 20 * np.log10(max_value) - 10 * np.log10(mse)


def jaccard_index(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate Jaccard Index (Intersection over Union) between two binary images.
    
    Parameters
    ----------
    img1 : np.ndarray
        First binary image.
    img2 : np.ndarray
        Second binary image.
        
    Returns
    -------
    float
        Jaccard Index.
    """
    intersection = np.logical_and(img1, img2).sum()
    union = np.logical_or(img1, img2).sum()
    
    if union == 0:
        return 1.0
    
    return intersection / union