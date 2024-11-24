import mediapipe as mp
import numpy as np
import cv2


class HandDetectionUtils:
    def __init__(self) -> None:
        self.handSolution = mp.solutions.hands
        self._hands = self.handSolution.Hands(min_detection_confidence=0.7, static_image_mode=False, max_num_hands=1)

    def process(self, frame):
        result = []
        recHands = self._hands.process(frame)
        if recHands.multi_hand_landmarks:
            return recHands.multi_hand_landmarks[0].landmark
        return None
    
    def draw_hand_landmarks(self, image, landmarks, height, width):
        for landmark in landmarks[:13]:
            # Convert landmark coordinates to pixel coordinates
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.circle(image, (x, y), 4, (0, 0, 255), -1)

        # Draw connections
        for connection in self.handSolution.HAND_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            if start_idx <= 12 and end_idx <= 12:
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                cv2.line(image, 
                        (int(start.x * width), int(start.y * height)), 
                        (int(end.x * width), int(end.y * height)), 
                        (0, 255, 0), 1)  # Draw connection line in green
    
    def draw_hand_landmarks_with_scale(self, image, landmarks, width, height, scale_factor):
        # Get landmark coordinates
        points = []
        for landmark in landmarks:
            points.append((landmark.x * width, landmark.y * height))

        # Calculate the bounding box of the hand
        x_coords, y_coords = zip(*points)
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))

        # Calculate the width and height of the bounding box
        box_width = x_max - x_min
        box_height = y_max - y_min

        # Determine scaling factor to fit the hand within 80% of the background size
        scale_factor = scale_factor * min(width / box_width, height / box_height)

        # Calculate the new scaled dimensions
        scaled_width = int(box_width * scale_factor)
        scaled_height = int(box_height * scale_factor)

        # Calculate the offset to center the scaled hand
        offset_x = (width - scaled_width) // 2
        offset_y = (height - scaled_height) // 2

        # Draw landmarks on the background
        for i, landmark in enumerate(landmarks[:13]):
            # Get normalized coordinates and scale them
            if i == 0:
                color = (255, 255, 255)
            elif i < 5:
                color = (255, 0 , 0)
            elif i < 9:
                color = (0, 255 , 0)
            else:
                color = (0, 0, 255)
            x = int((landmark.x * width - x_min) * scale_factor) + offset_x
            y = int((landmark.y * height - y_min) * scale_factor) + offset_y
            cv2.circle(image, (x, y), 3, color, -1)  # Draw landmark

        # Draw connections
        connections = self.handSolution.HAND_CONNECTIONS
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if start_idx <= 12 and end_idx <= 12:
                start = landmarks[start_idx]
                end = landmarks[end_idx]

                start_x = int((start.x * width - x_min) * scale_factor) + offset_x
                start_y = int((start.y * height - y_min) * scale_factor) + offset_y
                end_x = int((end.x * width - x_min) * scale_factor) + offset_x
                end_y = int((end.y * height - y_min) * scale_factor) + offset_y

                if end_idx < 5:
                    color = (255, 0 , 0)
                elif end_idx < 8:
                    color = (0, 255 , 0)
                elif end_idx < 13:
                    color = (0, 0, 255)
                
                cv2.line(image, (start_x, start_y), (end_x, end_y), color, 1)  # Draw line