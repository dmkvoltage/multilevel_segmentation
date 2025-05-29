"""
Segmentation Model

This module contains the main SegmentationModel class that implements
different segmentation strategies.
"""

import numpy as np
from typing import Literal, Optional, Tuple, Dict, Any

from .clustering import kmeans, fuzzy_cmeans
from .morphology import erosion, dilation, opening, closing
from .utils import normalize_image, convert_to_grayscale

class SegmentationModel:
    """
    A model for image segmentation using clustering and morphological operations.
    
    This model implements three approaches:
    1. Clustering-based segmentation
    2. Two-step approach (clustering followed by morphological filtering)
    3. Combined iterative approach (alternating between clustering and morphology)
    
    Parameters
    ----------
    method : str
        The segmentation method to use. Options are 'clustering', 'two_step', 
        or 'combined'.
    clusters : int
        The number of clusters to use.
    iterations : int
        The number of iterations for the combined approach.
    clustering_algorithm : str
        The clustering algorithm to use. Options are 'kmeans' or 'fuzzy_cmeans'.
    morphological_operations : list
        List of morphological operations to apply in sequence.
    """
    
    def __init__(
        self,
        method: Literal['clustering', 'two_step', 'combined'] = 'combined',
        clusters: int = 3,
        iterations: int = 5,
        clustering_algorithm: Literal['kmeans', 'fuzzy_cmeans'] = 'kmeans',
        morphological_operations: Optional[list] = None
    ):
        self.method = method
        self.clusters = clusters
        self.iterations = iterations
        self.clustering_algorithm = clustering_algorithm
        
        if morphological_operations is None:
            self.morphological_operations = ['opening', 'closing']
        else:
            self.morphological_operations = morphological_operations
            
        self._validate_params()
    
    def _validate_params(self):
        """Validate initialization parameters."""
        valid_methods = ['clustering', 'two_step', 'combined']
        if self.method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
            
        valid_clustering = ['kmeans', 'fuzzy_cmeans']
        if self.clustering_algorithm not in valid_clustering:
            raise ValueError(f"Clustering algorithm must be one of {valid_clustering}")
            
        valid_morphology = ['erosion', 'dilation', 'opening', 'closing']
        for op in self.morphological_operations:
            if op not in valid_morphology:
                raise ValueError(f"Morphological operation must be one of {valid_morphology}")
    
    def segment(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Segment the input image using the specified method.
        
        Parameters
        ----------
        image : np.ndarray
            The input image to segment.
            
        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary containing the segmentation results.
        """
        # Preprocess the image
        if len(image.shape) > 2 and image.shape[2] > 1:
            gray_image = convert_to_grayscale(image)
        else:
            gray_image = image.copy()
            
        normalized_image = normalize_image(gray_image)
        
        if self.method == 'clustering':
            return self._perform_clustering(normalized_image)
        elif self.method == 'two_step':
            return self._perform_two_step(normalized_image)
        elif self.method == 'combined':
            return self._perform_combined(normalized_image)
    
    def _perform_clustering(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Perform clustering-based segmentation."""
        if self.clustering_algorithm == 'kmeans':
            labels, centers = kmeans(image, self.clusters)
        else:  # fuzzy_cmeans
            labels, centers = fuzzy_cmeans(image, self.clusters)
            
        return {
            'segmented_image': self._create_segmented_image(labels, centers),
            'labels': labels,
            'centers': centers
        }
    
    def _perform_two_step(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Perform two-step segmentation (clustering followed by morphology)."""
        # Step 1: Clustering
        clustering_result = self._perform_clustering(image)
        segmented = clustering_result['segmented_image']
        
        # Step 2: Morphological filtering
        filtered = self._apply_morphological_operations(segmented)
        
        return {
            'segmented_image': filtered,
            'labels': clustering_result['labels'],
            'centers': clustering_result['centers'],
            'initial_segmentation': segmented
        }
    
    def _perform_combined(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Perform combined iterative segmentation."""
        current_image = image.copy()
        intermediate_results = []
        
        for i in range(self.iterations):
            # Perform clustering
            clustering_result = self._perform_clustering(current_image)
            segmented = clustering_result['segmented_image']
            intermediate_results.append(('clustering', segmented.copy()))
            
            # Apply morphological operations
            filtered = self._apply_morphological_operations(segmented)
            intermediate_results.append(('morphology', filtered.copy()))
            
            # Update current image for next iteration
            current_image = filtered
        
        return {
            'segmented_image': current_image,
            'intermediate_results': intermediate_results,
            'labels': clustering_result['labels'],
            'centers': clustering_result['centers']
        }
    
    def _apply_morphological_operations(self, image: np.ndarray) -> np.ndarray:
        """Apply a sequence of morphological operations."""
        result = image.copy()
        
        for op in self.morphological_operations:
            if op == 'erosion':
                result = erosion(result)
            elif op == 'dilation':
                result = dilation(result)
            elif op == 'opening':
                result = opening(result)
            elif op == 'closing':
                result = closing(result)
        
        return result
    
    def _create_segmented_image(self, labels: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Create a segmented image from cluster labels and centers."""
        segmented = np.zeros_like(labels, dtype=float)
        
        for i, center in enumerate(centers):
            segmented[labels == i] = center
        
        return segmented