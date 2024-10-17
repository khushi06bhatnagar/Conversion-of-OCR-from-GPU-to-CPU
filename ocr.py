#Here's the code for conversion of OCR from GPU to CPU with performance comparison.


!pip install opencv-python-headless easyocr matplotlib imutils

import cv2
import easyocr
import numpy as np
from matplotlib import pyplot as plt
import imutils
import time

# Function to process image with OCR and measure FPS
def process_image_with_ocr(image_path, use_gpu=False):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}.")
        return None, None

    # Resize image
    image = imutils.resize(image, width=500)

    # Initialize OCR reader (use GPU or CPU based on parameter)
    reader = easyocr.Reader(['en'], gpu=use_gpu)

    # Start timer for FPS measurement
    start_time = time.time()

    # Perform OCR
    results = reader.readtext(image)

    # Stop timer and calculate FPS
    end_time = time.time()
    processing_time = end_time - start_time
    fps = 1 / processing_time if processing_time > 0 else 0  # Avoid division by zero

    # Display results bounding boxes and recognized text
    for (bbox, text, prob) in results:
        # Draw bounding box
        (tl, tr, br, bl) = bbox
        tl = (int(tl[0]), int(tl[1]))
        br = (int(br[0]), int(br[1]))
        cv2.rectangle(image, tl, br, (0, 255, 0), 2)
        # Print recognized text and probability
        print(f"Text: {text}, Probability: {prob:.2f}")

    # Show image with bounding boxes
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"{'GPU' if use_gpu else 'CPU'} Mode - FPS: {fps:.2f}")
    plt.axis('off')  # Hide axes
    plt.show()

    return fps, results

# Specify the image path
image_path = "//content//ok.jpeg"  # Replace with your image path

# Process image using GPU
print("Processing with GPU...")
gpu_fps, gpu_results = process_image_with_ocr(image_path, use_gpu=True)

# Process image using CPU
print("Processing with CPU...")
cpu_fps, cpu_results = process_image_with_ocr(image_path, use_gpu=False)


# Check if gpu_fps and cpu_fps are not None before printing
if gpu_fps is not None and cpu_fps is not None:
    print(f"\nGPU FPS: {gpu_fps:.2f}, CPU FPS: {cpu_fps:.2f}")
else:
    print("Could not calculate FPS due to image loading issues.")

print("\nAccuracy Comparison:")
if gpu_results is not None and cpu_results is not None:
    for gpu_text, cpu_text in zip(gpu_results, cpu_results):
        print(f"GPU Text: {gpu_text[1]} | CPU Text: {cpu_text[1]}")
else:
    print("Could not retrieve OCR results due to processing issues.")
