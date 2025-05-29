"""
Morphological Operations

This module contains implementations of basic morphological operations
for image processing, built from scratch.
"""

import numpy as np
from typing import Tuple, Optional


def _create_structural_element(
    shape: str = 'square',
    size: int = 3
) -> np.ndarray:
    """
    Create a structural element (kernel) for morphological operations.
    
    Parameters
    ----------
    shape : str, optional
        Shape of the structural element. Options are 'square', 'cross', 
        'diamond'. Default is 'square'.
    size : int, optional
        Size of the structural element. Default is 3.
        
    Returns
    -------
    np.ndarray
        The structural element as a binary array.
    """
    if size % 2 == 0:
        size += 1  # Ensure odd size for centered element
        
    radius = size // 2
    center = (radius, radius)
    se = np.zeros((size, size), dtype=bool)
    
    if shape == 'square':
        se.fill(True)
    elif shape == 'cross':
        se[radius, :] = True
        se[:, radius] = True
    elif shape == 'diamond':
        for i in range(size):
            for j in range(size):
                if abs(i - center[0]) + abs(j - center[1]) <= radius:
                    se[i, j] = True
    else:
        raise ValueError("Shape must be one of 'square', 'cross', or 'diamond'")
        
    return se


def erosion(
    image: np.ndarray,
    se: Optional[np.ndarray] = None,
    se_shape: str = 'square',
    se_size: int = 3
) -> np.ndarray:
    """
    Perform morphological erosion on an image.
    
    Parameters
    ----------
    image : np.ndarray
        The input image.
    se : np.ndarray, optional
        The structural element. If None, it will be created using se_shape and se_size.
    se_shape : str, optional
        Shape of the structural element if se is None. Default is 'square'.
    se_size : int, optional
        Size of the structural element if se is None. Default is 3.
        
    Returns
    -------
    np.ndarray
        The eroded image.
    """
    if se is None:
        se = _create_structural_element(se_shape, se_size)
        
    # Convert image to binary if it's not already
    if image.dtype != bool:
        # Assuming values above 0.5 are foreground in normalized image
        binary_image = image > 0.5
    else:
        binary_image = image.copy()
    
    # Get dimensions
    height, width = binary_image.shape
    se_height, se_width = se.shape
    pad_height, pad_width = se_height // 2, se_width // 2
    
    # Create output image
    eroded = np.zeros_like(binary_image)
    
    # Perform erosion
    for i in range(pad_height, height - pad_height):
        for j in range(pad_width, width - pad_width):
            # Extract region of interest
            roi = binary_image[i - pad_height:i + pad_height + 1,
                              j - pad_width:j + pad_width + 1]
            
            # Apply structural element
            if np.all(roi[se] == True):
                eroded[i, j] = True
    
    # Convert back to original type if needed
    if image.dtype != bool:
        eroded = eroded.astype(image.dtype)
        if image.dtype == np.uint8:
            eroded = eroded * 255
            
    return eroded


def dilation(
    image: np.ndarray,
    se: Optional[np.ndarray] = None,
    se_shape: str = 'square',
    se_size: int = 3
) -> np.ndarray:
    """
    Perform morphological dilation on an image.
    
    Parameters
    ----------
    image : np.ndarray
        The input image.
    se : np.ndarray, optional
        The structural element. If None, it will be created using se_shape and se_size.
    se_shape : str, optional
        Shape of the structural element if se is None. Default is 'square'.
    se_size : int, optional
        Size of the structural element if se is None. Default is 3.
        
    Returns
    -------
    np.ndarray
        The dilated image.
    """
    if se is None:
        se = _create_structural_element(se_shape, se_size)
        
    # Convert image to binary if it's not already
    if image.dtype != bool:
        # Assuming values above 0.5 are foreground in normalized image
        binary_image = image > 0.5
    else:
        binary_image = image.copy()
    
    # Get dimensions
    height, width = binary_image.shape
    se_height, se_width = se.shape
    pad_height, pad_width = se_height // 2, se_width // 2
    
    # Create output image
    dilated = np.zeros_like(binary_image)
    
    # Perform dilation
    for i in range(pad_height, height - pad_height):
        for j in range(pad_width, width - pad_width):
            # Extract region of interest
            roi = binary_image[i - pad_height:i + pad_height + 1,
                              j - pad_width:j + pad_width + 1]
            
            # Apply structural element
            if np.any(roi[se] == True):
                dilated[i, j] = True
    
    # Convert back to original type if needed
    if image.dtype != bool:
        dilated = dilated.astype(image.dtype)
        if image.dtype == np.uint8:
            dilated = dilated * 255
            
    return dilated


def opening(
    image: np.ndarray,
    se: Optional[np.ndarray] = None,
    se_shape: str = 'square',
    se_size: int = 3
) -> np.ndarray:
    """
    Perform morphological opening on an image (erosion followed by dilation).
    
    Parameters
    ----------
    image : np.ndarray
        The input image.
    se : np.ndarray, optional
        The structural element. If None, it will be created using se_shape and se_size.
    se_shape : str, optional
        Shape of the structural element if se is None. Default is 'square'.
    se_size : int, optional
        Size of the structural element if se is None. Default is 3.
        
    Returns
    -------
    np.ndarray
        The opened image.
    """
    if se is None:
        se = _create_structural_element(se_shape, se_size)
        
    # Opening: erosion followed by dilation
    eroded = erosion(image, se)
    opened = dilation(eroded, se)
    
    return opened


def closing(
    image: np.ndarray,
    se: Optional[np.ndarray] = None,
    se_shape: str = 'square',
    se_size: int = 3
) -> np.ndarray:
    """
    Perform morphological closing on an image (dilation followed by erosion).
    
    Parameters
    ----------
    image : np.ndarray
        The input image.
    se : np.ndarray, optional
        The structural element. If None, it will be created using se_shape and se_size.
    se_shape : str, optional
        Shape of the structural element if se is None. Default is 'square'.
    se_size : int, optional
        Size of the structural element if se is None. Default is 3.
        
    Returns
    -------
    np.ndarray
        The closed image.
    """
    if se is None:
        se = _create_structural_element(se_shape, se_size)
        
    # Closing: dilation followed by erosion
    dilated = dilation(image, se)
    closed = erosion(dilated, se)
    
    return closed