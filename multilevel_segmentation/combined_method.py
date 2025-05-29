"""
Combined Morphological-Clustering Method

This module implements our novel iterative approach that combines
clustering and morphological operations for improved segmentation results.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from .clustering import kmeans, fuzzy_cmeans
from .morphology import erosion, dilation, opening, closing
from .utils import normalize_image


class CombinedMethod:
    """
    Combined Morphological-Clustering Method (CMCM) for image segmentation.
    
    This class implements our novel iterative approach that alternates between
    clustering and morphological operations, with each iteration refining the
    segmentation based on feedback from the previous step.
    
    Parameters
    ----------
    n_clusters : int, optional
        Number of clusters. Default is 3.
    n_iterations : int, optional
        Number of iterations. Default is 5.
    clustering_algorithm : str, optional
        Clustering algorithm to use. Options are 'kmeans' or 'fuzzy_cmeans'.
        Default is 'kmeans'.
    morphological_sequence : List[str], optional
        Sequence of morphological operations to apply in each iteration.
        Default is ['opening', 'closing'].
    feedback_strength : float, optional
        Strength of the feedback between iterations (0 to 1).
        Default is 0.5.
    """
    
    def __init__(
        self,
        n_clusters: int = 3,
        n_iterations: int = 5,
        clustering_algorithm: str = 'kmeans',
        morphological_sequence: Optional[List[str]] = None,
        feedback_strength: float = 0.5
    ):
        self.n_clusters = n_clusters
        self.n_iterations = n_iterations
        self.clustering_algorithm = clustering_algorithm
        
        if morphological_sequence is None:
            self.morphological_sequence = ['opening', 'closing']
        else:
            self.morphological_sequence = morphological_sequence
            
        self.feedback_strength = feedback_strength
        self._validate_params()
        
    def _validate_params(self):
        """Validate initialization parameters."""
        if self.n_clusters < 2:
            raise ValueError("Number of clusters must be at least 2")
            
        if self.n_iterations < 1:
            raise ValueError("Number of iterations must be at least 1")
            
        if self.clustering_algorithm not in ['kmeans', 'fuzzy_cmeans']:
            raise ValueError("Clustering algorithm must be 'kmeans' or 'fuzzy_cmeans'")
            
        valid_morphology = ['erosion', 'dilation', 'opening', 'closing']
        for op in self.morphological_sequence:
            if op not in valid_morphology:
                raise ValueError(f"Morphological operation must be one of {valid_morphology}")
                
        if not 0 <= self.feedback_strength <= 1:
            raise ValueError("Feedback strength must be between 0 and 1")
    
    def segment(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Segment the input image using the combined method.
        
        Parameters
        ----------
        image : np.ndarray
            Input image to segment.
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing segmentation results.
        """
        # Normalize the image
        normalized_image = normalize_image(image)
        
        # Initialize with clustering
        if self.clustering_algorithm == 'kmeans':
            labels, centers = kmeans(normalized_image, self.n_clusters)
        else:  # fuzzy_cmeans
            labels, centers = fuzzy_cmeans(normalized_image, self.n_clusters)
            
        # Create initial segmentation
        current_segmentation = self._create_segmented_image(labels, centers)
        
        # Store intermediate results
        intermediate_results = [('initial_clustering', current_segmentation.copy())]
        
        # Iterative refinement
        for i in range(self.n_iterations):
            # Apply morphological operations
            morphological_result = self._apply_morphological_sequence(current_segmentation)
            intermediate_results.append(('morphology', morphological_result.copy()))
            
            # Create weighted combination of original image and morphological result
            weighted_image = (1 - self.feedback_strength) * normalized_image + \
                             self.feedback_strength * morphological_result
            
            # Apply clustering to the weighted image
            if self.clustering_algorithm == 'kmeans':
                labels, centers = kmeans(weighted_image, self.n_clusters)
            else:  # fuzzy_cmeans
                labels, centers = fuzzy_cmeans(weighted_image, self.n_clusters)
                
            # Update current segmentation
            current_segmentation = self._create_segmented_image(labels, centers)
            intermediate_results.append(('clustering', current_segmentation.copy()))
            
        return {
            'segmented_image': current_segmentation,
            'labels': labels,
            'centers': centers,
            'intermediate_results': intermediate_results
        }
    
    def _apply_morphological_sequence(self, image: np.ndarray) -> np.ndarray:
        """Apply the sequence of morphological operations."""
        result = image.copy()
        
        for operation in self.morphological_sequence:
            if operation == 'erosion':
                result = erosion(result)
            elif operation == 'dilation':
                result = dilation(result)
            elif operation == 'opening':
                result = opening(result)
            elif operation == 'closing':
                result = closing(result)
                
        return result
    
    def _create_segmented_image(self, labels: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Create a segmented image from cluster labels and centers."""
        segmented = np.zeros_like(labels, dtype=float)
        
        for i, center in enumerate(centers):
            segmented[labels == i] = center
            
        return segmented