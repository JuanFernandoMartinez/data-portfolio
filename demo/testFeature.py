import pandas as pd
import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import flood_fill
from skimage import io, color, exposure, img_as_ubyte
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster import KMeans, MeanShift
from skimage.measure import label, regionprops
from scipy.stats import skew


# Images root directorie
data_dir = "cancerSeno_bw/test"
output_dir = "featuresTestFromTest"

# Class list
classes = ["sano", "cancer"]

# Create output directories
os.makedirs(os.path.join(output_dir, "sano"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "cancer"), exist_ok=True)

# Define angles and distanxes for GLCM (Gray-Level Co-Occurrence Matrix)
angles = [0]
distances = [1]

# Define normalization threshold
treshold = 120

# Define max number of clusters for segmentation
max_clusters = 5

# Initialize lists
# Store texture and density of pixels characteristics
glcm_features = []
densidades_pixeles_oscuros = []
num_cúmulos = []

# Store coordinates of origins in clusters
cluster_seeds = []

# Store loaded images
loaded_images = []


# Accumulate histograms, GLCM features, and dark pixel densities
histograms = {cls: [] for cls in classes}
glcm_features_all = {cls: [] for cls in classes}
dark_pixel_densities = {cls: [] for cls in classes}
brightness_values = {cls: [] for cls in classes}
std_deviation_values = {cls: [] for cls in classes}
skewness_values = {cls: [] for cls in classes}


for cls in classes:
    class_output_dir = os.path.join(output_dir, cls)
    os.makedirs(os.path.join(class_output_dir, "histogram"), exist_ok=True)
    os.makedirs(os.path.join(class_output_dir, "glcmMatrix"), exist_ok=True)
    os.makedirs(os.path.join(class_output_dir, "pixelDensity"), exist_ok=True)
    os.makedirs(os.path.join(class_output_dir, "brightness"), exist_ok=True)
    os.makedirs(os.path.join(class_output_dir, "stdDeviation"), exist_ok=True)
    os.makedirs(os.path.join(class_output_dir, "skewness"), exist_ok=True)
    os.makedirs(class_output_dir, exist_ok=True)


def get_images(class_dir):
    file_list = os.listdir(class_dir)
    images = [os.path.join(class_dir, image_name) for image_name in file_list]
    return images

for cls in classes:
    # Obtain route from current directory
    class_dir = os.path.join(data_dir, cls)
    images = get_images(class_dir)

    # Add image paths to the loaded-images list
    loaded_images.extend(images)
print("Images loaded")


def save_to_csv(data_list, output_path, column_name):
    """Save the data list to a CSV file."""
    df = pd.DataFrame(data_list, columns=[column_name])
    df.to_csv(output_path, index=False)


def calculate_dark_pixel_density(image, treshold):
    """Calculate the density of dark pixels in hyperchromatic areas of an image."""
    _, binary_image = cv2.threshold(image, treshold, 255, cv2.THRESH_BINARY)
    dark_pixels_count = np.sum(binary_image == 0)
    total_area = np.sum(binary_image == 255)
    dark_pixels_density = dark_pixels_count / total_area if total_area > 0 else 0
    return dark_pixels_density

def save_pixel_density_to_csv(pixel_densities, output_path):
    """Save the pixel densities list to a CSV file."""
    df = pd.DataFrame(pixel_densities, columns=['pixel_density'])
    df.to_csv(output_path, index=False)


for cls in classes:
    for image_path in images:
        image_name = os.path.basename(image_path).split('.')[0]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Dark pixel density
        dark_pixel_density = calculate_dark_pixel_density(image, treshold)
        dark_pixel_densities[cls].append(dark_pixel_density)
    print("first class generated")
print("second class generated")

# Save all dark pixel densities to a CSV file for each class
for cls in classes:
    combined_pixel_density_output_path = os.path.join(output_dir, cls, "pixelDensity", "combined_pixelDensity.csv")
    save_pixel_density_to_csv(dark_pixel_densities[cls], combined_pixel_density_output_path)
print("pixel densities saved")


def calculate_brightness(image):
    """Calculate the brightness of an image."""
    return np.mean(image)


for cls in classes:
    for image_path in images:
        image_name = os.path.basename(image_path).split('.')[0]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Brightness
        brightness = calculate_brightness(image)
        brightness_values[cls].append(brightness)
    print("first class generated")
print("second class generated")

# Save all brightness values to a CSV file for each class
for cls in classes:
    combined_brightness_output_path = os.path.join(output_dir, cls, "brightness", "combined_brightness.csv")
    save_to_csv(brightness_values[cls], combined_brightness_output_path, "brightness")
print("brightness saved")


def calculate_std_deviation(image):
    """Calculate the standard deviation of an image."""
    return np.std(image)


for cls in classes:
    for image_path in images:
        image_name = os.path.basename(image_path).split('.')[0]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Standard deviation
        std_deviation = calculate_std_deviation(image)
        std_deviation_values[cls].append(std_deviation)
    print("first class generated")
print("second class generated")
                                         
# Save all standard deviation values to a CSV file for each class
for cls in classes:
    combined_std_deviation_output_path = os.path.join(output_dir, cls, "stdDeviation", "combined_stdDeviation.csv")
    save_to_csv(std_deviation_values[cls], combined_std_deviation_output_path, "std_deviation")
print("std deviation saved")


def calculate_skewness(image):
    """Calculate the skewness of an image."""
    return skew(image.flatten())

for cls in classes:
    for image_path in images:
        image_name = os.path.basename(image_path).split('.')[0]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Skewness
        skewness = calculate_skewness(image)
        skewness_values[cls].append(skewness)
    print("first class generated")
print("second class generated")


for cls in classes:
    combined_skewness_output_path = os.path.join(output_dir, cls, "skewness", "combined_skewness.csv")
    save_to_csv(skewness_values[cls], combined_skewness_output_path, "skewness")
print("skewness saved")