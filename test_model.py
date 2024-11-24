import cv2
import copy
import numpy as np
import tensorflow as tf
from IPython.display import clear_output
from utils.CameraUtils import CameraUtils
from utils.HandDetectionUtils import HandDetectionUtils

height, width = 100, 100
fps = 30
handDetectionUtils = HandDetectionUtils()
cameraUtils = CameraUtils(height, width, fps)
skip_frames = 3
end = False

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path='models/model.tflite')
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

frame_count = 0
while True:
    frame_count = frame_count + 1
    landmark_frame = np.ones((height, width, 3), dtype=np.uint8) * 0
    if end == True:
        break
            
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
    
    # Control
    if wait_key == ord('q'):
        end = True
        break
            
    landmark_frame = cv2.cvtColor(landmark_frame, cv2.COLOR_BGR2RGB)
    predicted_frame = np.array(landmark_frame).astype('float32') / 255.0
    predicted_frame = predicted_frame.reshape(-1, height, width, 3)
    
    # Handle queue
    if landmarks and frame_count % skip_frames == 0:
        # Set the input tensor to the preprocessed frame
        input_data = np.array(predicted_frame, dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Run inference
        interpreter.invoke()

        # Get the output tensor and post-process
        prediction = interpreter.get_tensor(output_details[0]['index'])
        predicted_class = np.argmax(prediction, axis=1)
        
        clear_output(False)
        print(f'class: {predicted_class}')

cameraUtils.close()
