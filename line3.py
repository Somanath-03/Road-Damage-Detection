import numpy as np
import cv2
import math

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                if 35 <= angle <= 115 or -115 <= angle <= -35:
                    cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    return cv2.addWeighted(initial_img, α, img, β, γ)

def lane_finding_pipeline(image):
    gray = grayscale(image)
    blur = gaussian_blur(gray, 5)
    edges = canny(blur, 50, 150)
    imshape = image.shape
    vertices = np.array([[(0, imshape[0]), (imshape[1] // 4, imshape[0] - (imshape[0] // 3)), (3 * imshape[1] // 4, imshape[0] - (imshape[0] // 3)), (imshape[1], imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)
    line_image = hough_lines(masked_edges, 2, np.pi / 180, 15, 40, 20)
    result = weighted_img(line_image, image)
    return result

def process_video(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_video_path}")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video properties: width={frame_width}, height={frame_height}, frame_rate={frame_rate}, total_frames={total_frames}")

    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (frame_width, frame_height))

    if not out.isOpened():
        print(f"Error: Could not open video writer for file {output_video_path}")
        return

    try:
        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                print(f"Error: Could not read frame {frame_idx}")
                break
            frame = cv2.resize(frame, (640, 480))
            processed_frame = lane_finding_pipeline(frame)
            out.write(processed_frame)
            cv2.imshow('Processed Frame', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("Video processing complete. Processed video saved as:", output_video_path)

# Replace 'input_video.mp4' with the path to your video file
input_video_path = r"./videos/8358-208052058_small.mp4"
output_video_path = r'./videos/processed_video2.mp4'
process_video(input_video_path, output_video_path)
