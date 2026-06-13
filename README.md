# Multi-Scale Object Detection using YOLO-MS and Diffusion-Based Denoising

## Overview

This project presents a Multi-Scale Object Detection system that combines YOLO-MS with Diffusion-Based Denoising techniques to improve object detection accuracy in noisy and complex environments. The diffusion model enhances image quality by reducing noise, while YOLO-MS detects objects of different scales efficiently.

## Features

* Multi-scale object detection
* Noise reduction using diffusion-based denoising
* Real-time object recognition
* Web-based interface using Flask
* Support for multiple object classes

## Technologies Used

* Python
* OpenCV
* YOLO
* Flask
* NumPy

## Project Structure

* `app.py` - Main application file
* `yolo.py` - Object detection logic
* `yolo.cfg` - YOLO configuration file
* `templates/` - HTML templates
* `static/` - CSS, JavaScript, and static assets
* `test_images/` - Sample images for testing

## How to Run

1. Install the required dependencies.
2. Place the YOLO weights file in the project directory.
3. Run:

```bash
python app.py
```

4. Open the application in your browser.

## Results

The proposed approach improves object detection performance in noisy images by integrating diffusion-based denoising with multi-scale feature extraction.

