import os
import cv2
import base64
import numpy as np
import math
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from inference_sdk import InferenceHTTPClient
import os

API_KEY = os.getenv("SECRET_KEY")
API_URL = os.getenv("DATABASE_URL")

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create an inference client
CLIENT = InferenceHTTPClient(
    api_url= API_URL,
    api_key= API_KEY
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_potholes', methods=['POST'])
def detect_potholes():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    video = request.files['video']
    if video.filename == '':
        return jsonify({'error': 'No video file selected'}), 400
    
    if video:
        filename = secure_filename(video.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video.save(video_path)
        print("Video Received")
        pothole_frames = []
        frame_count = 0
        
        cap = cv2.VideoCapture(video_path)
        print("Processing started")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every 60th frame
            if frame_count % 60 != 0:
                continue
            
            # Lane detection pipeline
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 50, 150)
            imshape = frame.shape
            vertices = np.array([[(0, imshape[0]), (imshape[1] // 4, imshape[0] - (imshape[0] // 3)), 
                                  (3 * imshape[1] // 4, imshape[0] - (imshape[0] // 3)), (imshape[1], imshape[0])]], dtype=np.int32)
            mask = np.zeros_like(edges)
            cv2.fillPoly(mask, vertices, 255)
            masked_edges = cv2.bitwise_and(edges, mask)
            lines = cv2.HoughLinesP(masked_edges, 2, np.pi / 180, 15, np.array([]), minLineLength=40, maxLineGap=20)
            line_img = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
            if lines is not None:
                for line in lines:
                    for x1, y1, x2, y2 in line:
                        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                        if 35 <= angle <= 115 or -115 <= angle <= -35:
                            cv2.line(line_img, (x1, y1), (x2, y2), [255, 0, 0], 2)
            processed_frame = cv2.addWeighted(frame, 0.8, line_img, 1., 0.)
            
            # Save frame to a temporary file
            frame_path = os.path.join(app.config['UPLOAD_FOLDER'], f'frame_{frame_count}.jpg')
            cv2.imwrite(frame_path, processed_frame)
            
            # Run inference using the API
            response = CLIENT.infer(frame_path, model_id="pothole-abpqz-utfx2/1")
            results = response['predictions']
            
            for result in results:
                if result['confidence'] > 0.4:
                    # get coordinates
                    x1, y1, x2, y2 = result['x'], result['y'], result['x'] + result['width'], result['y'] + result['height']
                    # convert to int
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # get the class
                    cls = result['class']

                    # get the respective colour
                    colour = (0, 255, 0)  # Green

                    # draw the rectangle
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), colour, 2)

                    # put the class name and confidence on the image
                    cv2.putText(processed_frame, f'{cls} {result["confidence"]:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)
                    # Convert frame to JPEG
                    _, buffer = cv2.imencode('.jpg', processed_frame)
                    # Convert to base64 string
                    img_str = base64.b64encode(buffer).decode('utf-8')
                    pothole_frames.append({
                        'frame_number': frame_count,
                        'image': img_str
                    })
                    break  # Only need one detection per frame
        
        cap.release()
        os.remove(video_path)  # Clean up the uploaded file
        
        return jsonify({
            'total_frames': frame_count,
            'pothole_frames': pothole_frames
        })

if __name__ == '__main__':
    app.run(debug=True)