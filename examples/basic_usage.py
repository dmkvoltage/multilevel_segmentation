"""
Basic Usage Example

This example demonstrates the basic usage of the multilevel_segmentation library.
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

from multilevel_segmentation import SegmentationModel
from multilevel_segmentation.visualization import visualize_results, save_all_results

# Create directories if they don't exist
os.makedirs('images/input', exist_ok=True)
os.makedirs('images/output', exist_ok=True)

# Load an image
try:
    # Try to load from images/input directory
    image_files = [f for f in os.listdir('images/input') 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if image_files:
        image_path = os.path.join('images/input', image_files[0])
        img = np.array(Image.open(image_path))
        print(f"Loaded image: {image_path}")
    else:
        # Create a synthetic image if no images are available
        print("No images found in images/input/, creating synthetic image...")
        size = 200
        img = np.zeros((size, size), dtype=np.float32)
        
        # Create some shapes for segmentation
        # Circle
        for i in range(size):
            for j in range(size):
                if (i - size//2)**2 + (j - size//2)**2 < (size//4)**2:
                    img[i, j] = 0.8
        
        # Rectangle
        img[size//6:2*size//6, size//6:5*size//6] = 0.5
        
        # Add noise
        np.random.seed(42)
        img += 0.1 * np.random.randn(size, size)
        img = np.clip(img, 0, 1)
        
        # Save synthetic image
        plt.imsave('images/input/synthetic_image.png', img, cmap='gray')
        print("Saved synthetic image to images/input/synthetic_image.png")

    # Create segmentation models
    print("\nPerforming segmentation...")
    
    # Clustering only
    clustering_model = SegmentationModel(method='clustering', clusters=3)
    clustering_result = clustering_model.segment(img)
    
    # Two-step method
    two_step_model = SegmentationModel(method='two_step', clusters=3)
    two_step_result = two_step_model.segment(img)
    
    # Combined method
    combined_model = SegmentationModel(method='combined', clusters=3, iterations=5)
    combined_result = combined_model.segment(img)
    
    # Prepare results dictionary
    results = {
        'clustered_image': clustering_result['segmented_image'],
        'morphological_image': two_step_result['segmented_image'],
        'combined_image': combined_result['segmented_image']
    }
    
    # Visualize and save results
    print("\nSaving results...")
    
    # Save comparison visualization
    visualize_results(
        img, 
        results,
        save_path='images/output/segmentation_comparison.png',
        show=True
    )
    
    # Save individual results
    save_all_results(
        img,
        results,
        output_dir='images/output'
    )
    
    print("\nResults saved to images/output/")
    print("- segmentation_comparison.png: Combined visualization")
    print("- original.png: Original image")
    print("- clustered.png: Clustering result")
    print("- morphological.png: Two-step result")
    print("- combined.png: Combined method result")

except Exception as e:
    print(f"Error: {e}")