# Pothole Detection System

A computer vision system for detecting potholes in road videos using YOLOv8 object detection and lane detection algorithms.

## Project Overview

This project combines pothole detection with lane detection to create a comprehensive road monitoring system. It includes:

1. A trained YOLOv8 model for pothole detection
2. Lane detection algorithms to identify road boundaries
3. Real-time video processing capabilities
4. A web interface for uploading and analyzing videos

## Components

### Core Detection Modules

- **main.py**: Real-time pothole detection using webcam feed
- **line3.py**: Lane detection pipeline for road videos

### Web Application

- **app.py**: Flask web application with local YOLOv8 model
- **app_w_api.py**: Flask web application using Roboflow API for inference

### Model Training

- **runs/detect/train4/args.yaml**: Configuration for the YOLOv8 model training

## Features

- Real-time pothole detection
- Lane line identification
- Video processing with frame sampling
- Web interface for uploading videos
- Visual highlighting of detected potholes
- API integration option for cloud-based inference

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pothole-detection.git
cd pothole-detection

# Install dependencies
pip install -r requirements.txt

# Download the trained model (if not included)
# Place model1.pt or best.pt in the project root
```

## Usage

### Real-time Detection

```bash
python main.py
```

### Process a Video File with Lane Detection

```bash
python line3.py
```

### Run the Web Application

```bash
# Using local model
python app.py

# Using Roboflow API (requires API keys)
export API_KEY="your_api_key"
export API_URL="your_api_url"
python app_w_api.py
```

Then navigate to `http://localhost:5000` in your web browser.

## Model Training

The model was trained using YOLOv8 with the following configuration:

- Base model: YOLOv8s
- Epochs: 10
- Batch size: 16
- Image size: 640x640
- Augmentation: Enabled (flips, mosaic, HSV adjustments)

For full training details, see `runs/detect/train4/args.yaml`.

## Technologies Used

- YOLOv8 for object detection
- OpenCV for image processing
- Flask for web application
- Roboflow for API integration (optional)
- NumPy for numerical operations

## License



## Acknowledgments

- Ultralytics for YOLOv8
- Roboflow for inference API
## Contributors
- Somanath Meda
- Vishal S
- Vaibhav VB
