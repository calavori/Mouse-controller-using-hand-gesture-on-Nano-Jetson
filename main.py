import time
import cv2
import math
import copy
import keyboard
import numpy as np
import pyautogui
import tensorflow as tf
from enum import Enum
from utils.CameraUtils import CameraUtils
from utils.HandDetectionUtils import HandDetectionUtils

pyautogui.FAILSAFE = False

class ACTIONS(Enum):
    NONE='none'
    CLICK='click'
    DOUBLE_CLICK='double click'
    RIGHT_CLICK='right click'
    DRAG='drag'
    RELEASE='release drag'
    UP='swept up'
    DOWN='swept down'
    LEFT='swept left'
    RIGHT='swept right'
    SWITCH='on/off movement'

class SmoothFilter:
    def __init__(self):
        self.last_x = None
        self.last_y = None

    def update(self, x, y):
        if self.last_x is None or self.last_y is None:
            self.last_x, self.last_y = x, y
        else:
            delta_x = x - self.last_x
            delta_y = y - self.last_y
            dist = math.sqrt(delta_x**2 + delta_y**2)
            if dist <= 25:
                alpha = 0
            elif dist <= 80:
                alpha = 0.1
            elif dist <= 200:
                alpha = 0.3
            elif dist <= 900:
                alpha = 0.8
            else:
                alpha = 1
            self.last_x = self.last_x + delta_x * alpha
            self.last_y = self.last_y + delta_y * alpha
        return self.last_x, self.last_y

def add_predicted_data(predicted_arr, predicted_value):
    if len(predicted_arr) == 3:
        predicted_arr.pop(0)
    predicted_arr.append(predicted_value)

def classify_dynamic_action(predicted_arr):
    if len(predicted_arr) != 3:
        return ACTIONS.NONE
    code = '-'.join(predicted_arr)
    if code == '0-2-0':
        return ACTIONS.CLICK
    if code == '0-2-2':
        return ACTIONS.DOUBLE_CLICK
    elif code == '0-3-0':
        return ACTIONS.RIGHT_CLICK
    elif code == '1-1-1':
        return ACTIONS.DRAG
    elif code == '5-4-4': 
        return ACTIONS.UP
    elif code == '4-5-5': 
        return ACTIONS.DOWN
    elif code == '7-6-6': 
        return ACTIONS.LEFT
    elif code == '6-7-7': 
        return ACTIONS.RIGHT
    elif code == '0-8-8': 
        return ACTIONS.SWITCH
    else:
        return ACTIONS.NONE
    
def handle_gesture(action):
    # if ACTIONS != ACTIONS.NONE:
    #     print(action)
    if action == ACTIONS.CLICK:
        pyautogui.click()
    elif action == ACTIONS.DOUBLE_CLICK:
        pyautogui.click(clicks=2)
    elif action == ACTIONS.RIGHT_CLICK:
        pyautogui.rightClick()
    elif action == ACTIONS.DRAG:
        pyautogui.mouseDown()
    elif action == ACTIONS.RELEASE:
        pyautogui.mouseUp()
    elif action == ACTIONS.UP:
        pyautogui.hscroll(-500)
    elif action == ACTIONS.DOWN:
        pyautogui.hscroll(500)
    # elif action == ACTIONS.LEFT:
    #     keyboard.press("left")
    # elif action == ACTIONS.RIGHT:
    #     keyboard.press("right")
    
def reset_control_zone(frame_width, frame_height, hand_center_x, hand_center_y, size_percent=0.4):
    control_zone_height = int(frame_height * size_percent)
    control_zone_width = int(frame_height * size_percent * pyautogui.size().width / pyautogui.size().height)

    control_zone_x = int(hand_center_x * frame_width - control_zone_width / 2)
    control_zone_y = int(hand_center_y * frame_height - control_zone_height / 2)

    pyautogui.moveTo(pyautogui.size().width / 2, pyautogui.size().height / 2)

    return {
        'startX': control_zone_x,
        'startY': control_zone_y,
        'width': control_zone_width,
        'height': control_zone_height
    }

def move_mouse_to(landmarkX, landmarkY, control_zone, frame_width, frame_height, smooth_filter):
    landmarkX = 1 - landmarkX
    if control_zone == None:
        return
    
    controlX = landmarkX * frame_width
    controlY = landmarkY * frame_height

    x = (controlX - control_zone['startX']) * pyautogui.size().width / control_zone['width']
    y = (controlY - control_zone['startY']) * pyautogui.size().height / control_zone['height']

    smoothed_x, smoothed_y = smooth_filter.update(x, y)
    pyautogui.moveTo(smoothed_x, smoothed_y)

def add_predicted_value(image, value):
    shown_text = f'Control: {value}'
    height, width = image.shape[:2]
    
    rect_width = int(width * 0.4)
    rect_height = int(height * 0.1)
    
    top_left = (0, 0)
    bottom_right = (rect_width, rect_height)
    
    cv2.rectangle(image, top_left, bottom_right, (255, 255, 255), thickness=cv2.FILLED)
    
    font = cv2.FONT_HERSHEY_TRIPLEX
    font_scale = 0.5
    thickness = 1
    text_size = cv2.getTextSize(shown_text, font, font_scale, thickness)[0]
    
    text_x = top_left[0] + (rect_width - text_size[0]) // 2
    text_y = top_left[1] + (rect_height + text_size[1]) // 2

    cv2.putText(image, shown_text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

    return image
    

def main():
    height, width = 100, 100
    fps = 30
    handDetectionUtils = HandDetectionUtils()
    cameraUtils = CameraUtils(height, width, fps)
    skip_frames = 7
    end = False
    interpreter = tf.lite.Interpreter(model_path='models/model.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    predicted_arr = []
    frame_count = 0
    control_zone = None
    smooth_filter = None
    on_drag = False
    last_action = ACTIONS.NONE.value
    prev_time = time.time()
    while True:
        frame_count = frame_count + 1
        landmark_frame = np.ones((height, width, 3), dtype=np.uint8) * 0
        if end == True:
            break
                
        # Get image
        original_frame = cameraUtils.read()
        frame_height, frame_width, _ = original_frame.shape
        shown_frame = copy.deepcopy(original_frame)
        # Process image
        landmarks = handDetectionUtils.process(shown_frame)
        # Display image
        shown_frame = cv2.flip(shown_frame, 1)
        add_predicted_value(shown_frame, last_action)
        shown_frame = cv2.flip(shown_frame, 1)
        if landmarks:
            handDetectionUtils.draw_hand_landmarks(shown_frame, landmarks, cameraUtils.frame_height, cameraUtils.frame_width)
            handDetectionUtils.draw_hand_landmarks_with_scale(landmark_frame, landmarks, cameraUtils.saved_height, cameraUtils.saved_width, 0.8)
            if control_zone != None:
                cv2.rectangle(shown_frame, 
                    (control_zone['startX'], control_zone['startY']), 
                    (control_zone['startX'] + control_zone['width'], control_zone['startY'] + control_zone['height']), 
                    (0, 255, 0),
                    thickness=2)     
                pointer_x = int(landmarks[5].x * frame_width)
                pointer_y = int(landmarks[5].y * frame_height)
                cv2.circle(shown_frame, (pointer_x, pointer_y), 5, (255, 0 , 0), -1)
            else:
                smooth_filter = None
        
        curr_time = time.time()
        fps_display = int(1 / (curr_time - prev_time))
        prev_time = curr_time
        fps_text = f'FPS: {fps_display}'
        text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = frame_width - text_size[0] - 10
        text_y = 30
        shown_frame = cv2.flip(shown_frame, 1)
        cv2.putText(shown_frame, fps_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        shown_frame = cv2.flip(shown_frame, 1)
        wait_key = cameraUtils.show(shown_frame, 'camera')

        # Control
        if wait_key == ord('q'):
            end = True
            break
                
        landmark_frame = cv2.cvtColor(landmark_frame, cv2.COLOR_BGR2RGB)
        predicted_frame = np.array(landmark_frame).astype('float32') / 255.0
        predicted_frame = predicted_frame.reshape(-1, height, width, 3)
        # Handle gestures
        if landmarks and frame_count % skip_frames == 0:
            input_data = np.array(predicted_frame, dtype=np.float32)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()

            prediction = interpreter.get_tensor(output_details[0]['index'])
            predicted_class = np.argmax(prediction, axis=1)
            add_predicted_data(predicted_arr, str(predicted_class[0]))

            action = classify_dynamic_action(predicted_arr)
            if action == ACTIONS.SWITCH:
                if control_zone == None:
                    control_zone = reset_control_zone(frame_width, frame_height, landmarks[5].x, landmarks[5].y, 0.4)
                else:
                    control_zone = None
            else:
                if control_zone:
                    if smooth_filter == None:
                        smooth_filter = SmoothFilter()
                    move_mouse_to(landmarks[5].x, landmarks[5].y, control_zone, frame_width, frame_height, smooth_filter)
                if action == ACTIONS.DRAG:
                    if on_drag == False:
                        on_drag = True
                    else:
                        continue
                if action == ACTIONS.NONE and on_drag == True:
                    on_drag = False
                    action = ACTIONS.RELEASE
                handle_gesture(action)
            last_action = action.value
                
    cameraUtils.close()


if __name__ == "__main__":
    main()