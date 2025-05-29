"""
Real-World Example

This example demonstrates the application of the multilevel_segmentation library
to real-world image segmentation tasks like medical imaging or satellite imagery.
"""

import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlretrieve
from PIL import Image
import os

from multilevel_segmentation import SegmentationModel
from multilevel_segmentation.combined_method import CombinedMethod
from multilevel_segmentation.visualization import visualize_results, compare_methods

# Create a directory for sample images
os.makedirs('sample_images', exist_ok=True)

# Download sample images
sample_images = {
    'Brain MRI': 'https://raw.githubusercontent.com/loli/medpy/master/examples/data/mr_t1.png',
    'Satellite': 'https://i.imgur.com/vGIL8DD.jpg',  # Aerial image
    'Cell': 'https://i.imgur.com/2TJ6IwV.jpg'  # Cell microscopy
}

# Load images
loaded_images = {}
for name, url in sample_images.items():
    # Define local path
    local_path = f"sample_images/{name.lower().replace(' ', '_')}.jpg"
    
    # Download if needed
    if not os.path.exists(local_path):
        print(f"Downloading {name} image...")
        try:
            urlretrieve(url, local_path)
            print(f"  Downloaded to {local_path}")
        except Exception as e:
            print(f"  Error downloading {name} image: {e}")
            continue
    
    # Load image
    try:
        img = np.array(Image.open(local_path).convert('L'))  # Convert to grayscale
        loaded_images[name] = img
        print(f"Loaded {name} image with shape {img.shape}")
    except Exception as e:
        print(f"  Error loading {name} image: {e}")

# Process each image
for name, img in loaded_images.items():
    print(f"\nProcessing {name} image...")
    
    # Normalize image to [0, 1]
    img_normalized = img / 255.0 if img.max() > 1 else img
    
    # Create different models based on image type
    if name == 'Brain MRI':
        # For medical images, use Fuzzy C-means with more clusters
        models = {
            'K-means': SegmentationModel(method='clustering', clustering_algorithm='kmeans', clusters=3),
            'Fuzzy C-means': SegmentationModel(method='clustering', clustering_algorithm='fuzzy_cmeans', clusters=4),
            'Two-Step': SegmentationModel(method='two_step', clustering_algorithm='fuzzy_cmeans', clusters=4),
            'CMCM': CombinedMethod(n_clusters=4, clustering_algorithm='fuzzy_cmeans', 
                                   morphological_sequence=['opening', 'closing'],
                                   feedback_strength=0.6, n_iterations=7)
        }
    elif name == 'Satellite':
        # For satellite images, use more iterations and custom morphological sequence
        models = {
            'K-means': SegmentationModel(method='clustering', clustering_algorithm='kmeans', clusters=5),
            'Two-Step': SegmentationModel(method='two_step', clusters=5),
            'CMCM-Light': CombinedMethod(n_clusters=5, morphological_sequence=['opening'], 
                                        feedback_strength=0.4, n_iterations=3),
            'CMCM-Full': CombinedMethod(n_clusters=5, morphological_sequence=['opening', 'closing'],
                                       feedback_strength=0.6, n_iterations=7)
        }
    else:  # Cell or default
        # For microscopy, use more emphasis on morphology
        models = {
            'K-means': SegmentationModel(method='clustering', clustering_algorithm='kmeans', clusters=3),
            'Two-Step': SegmentationModel(method='two_step', clusters=3),
            'CMCM-Weak': CombinedMethod(n_clusters=3, feedback_strength=0.3, n_iterations=5),
            'CMCM-Strong': CombinedMethod(n_clusters=3, feedback_strength=0.8, n_iterations=5)
        }
    
    # Run each model
    results = {}
    for model_name, model in models.items():
        print(f"  Running {model_name}...")
        result = model.segment(img_normalized)
        results[model_name] = result
    
    # Visualize comparison
    plt.figure(figsize=(15, 10))
    
    # Original image
    plt.subplot(2, 3, 1)
    plt.imshow(img_normalized, cmap='gray')
    plt.title(f"Original {name} Image")
    plt.axis('off')
    
    # Results for each model
    for i, (model_name, result) in enumerate(results.items(), 1):
        plt.subplot(2, 3, i + 1)
        plt.imshow(result['segmented_image'], cmap='viridis')
        plt.title(model_name)
        plt.axis('off')
        
        # If it's a combined method, also save the intermediate results
        if 'CMCM' in model_name and 'intermediate_results' in result:
            # Save a separate figure for intermediate results
            plt.figure(figsize=(15, 5))
            
            # Show original and every second intermediate result
            n_results = min(5, len(result['intermediate_results']))
            plt.subplot(1, n_results + 1, 1)
            plt.imshow(img_normalized, cmap='gray')
            plt.title(f"Original")
            plt.axis('off')
            
            for j in range(n_results):
                idx = j * (len(result['intermediate_results']) // n_results)
                step_name, step_img = result['intermediate_results'][idx]
                
                plt.subplot(1, n_results + 1, j + 2)
                plt.imshow(step_img, cmap='viridis')
                plt.title(f"Step {idx+1}: {step_name}")
                plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(f"{name.lower().replace(' ', '_')}_intermediate_{model_name}.png")
            plt.close()
    
    plt.tight_layout()
    plt.savefig(f"{name.lower().replace(' ', '_')}_comparison.png")
    plt.close()

print("\nReal-world examples completed! Results saved as PNG files.")