This repository demonstrates a comparison of Optical Character Recognition (OCR) performance on both GPU and CPU using the EasyOCR and OpenCV libraries. The goal of this project is to analyze how processing speeds and accuracy differ when utilizing GPU vs CPU for OCR tasks.

Overview
In this project, we process the same image using both GPU and CPU and measure:

FPS (Frames Per Second): To determine processing speed.
Accuracy: To ensure the text recognition is consistent across both hardware options.
The CPU performs better in terms of FPS for this specific task, while maintaining similar accuracy as the GPU.

Key Results
GPU FPS: 0.16
CPU FPS: 0.31
The CPU outperforms the GPU for this particular OCR task in terms of FPS, with both maintaining similar text recognition accuracy.# Conversion-of-OCR-from-GPU-to-CPU
This repository contains a Python script to compare the performance of Optical Character Recognition (OCR) tasks using both GPU and CPU, utilizing the EasyOCR and OpenCV libraries. 

Installation
To run this project, ensure you have the following libraries installed:


pip install opencv-python-headless easyocr matplotlib imutils
Usage
Download the code: Clone or download this repository to your local machine.
Run the script: You can process an image by running the ocr_comparison.py script. Make sure to specify the correct path to your image file.

python ocr_comparison.py
Example usage in code

image_path = "your_image.jpg"
gpu_fps, gpu_results = process_image_with_ocr(image_path, use_gpu=True)  # GPU mode
cpu_fps, cpu_results = process_image_with_ocr(image_path, use_gpu=False) # CPU mode
Compare Results: After processing, the script will output FPS for both GPU and CPU, as well as the recognized text and accuracy.

Key Features
GPU vs CPU Performance Comparison: Measure and compare the FPS between GPU and CPU during OCR processing.
OCR Text Recognition: Uses EasyOCR to detect and recognize text in images.
Bounding Box Visualization: Draws bounding boxes around recognized text and displays the image with bounding boxes using Matplotlib.
Accuracy and Confidence Reporting: The recognized text and its confidence score are printed in the console.
Example Output
yaml

Processing with GPU...
Text: Hello World, Probability: 0.98
GPU FPS: 0.16

Processing with CPU...
Text: Hello World, Probability: 0.98
CPU FPS: 0.31

Technologies Used
OpenCV: For image processing and loading.
EasyOCR: For text detection and recognition.
Matplotlib: For visualizing images and bounding boxes.
Imutils: For resizing images.

Future Enhancements
Optimization Techniques: Implement additional techniques like model quantization and pruning to further improve CPU performance.
Real-Time OCR: Expand the project to handle real-time video streams for more complex OCR tasks.
