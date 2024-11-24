import copy
import numpy as np
from IPython.display import clear_output
from datetime import datetime
from utils.CameraUtils import CameraUtils
from utils.HandDetectionUtils import HandDetectionUtils

label = '0'
height, width = 100, 100
handDetectionUtils = HandDetectionUtils()
cameraUtils = CameraUtils(width, height, 30)
image_count = 0

while True:
    landmark_frame = np.ones((height, width, 3), dtype=np.uint8) * 0
            
    # Get image
    original_frame = cameraUtils.read()
    shown_frame = copy.deepcopy(original_frame)
    # Process image
    landmarks = handDetectionUtils.process(shown_frame)
    # Display image
    if landmarks:
        handDetectionUtils.draw_hand_landmarks(shown_frame, landmarks, cameraUtils.frame_height, cameraUtils.frame_width)
        handDetectionUtils.draw_hand_landmarks_with_scale(landmark_frame, landmarks, cameraUtils.saved_height, cameraUtils.saved_width, 0.8)
        
    wait_key = cameraUtils.show(shown_frame, 'camera')
    cameraUtils.show(landmark_frame, 'hand')
    # Control
    if wait_key == ord('q'):
        cameraUtils.close()
        break
    elif wait_key == ord('s'):
        clear_output(False)
        output_path = f'resources/{label}/{datetime.now().strftime("%m%d%Y%H%M%S")}.jpg'
        cameraUtils.writeOutput(output_path, landmark_frame)
        image_count = image_count + 1
        print(f'count" {image_count}')
    elif 48 <= wait_key <= 57:
        image_count = 0
        label = chr(wait_key)
        clear_output(False)
        print(f'Set label to {label}')