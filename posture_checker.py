import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks.python.vision import PoseLandmarker
from mediapipe.tasks.python.vision.core import VisionRunningMode

class PostureChecker:
    def __init__(self, model_path, knee_threshold=160, ankle_height_threshold=0.01):
        self.knee_threshold = knee_threshold  # Angle threshold for knees to be considered bent (in degrees)
        self.ankle_height_threshold = ankle_height_threshold  # Threshold for detecting if the ankle is on the ground
        
        # Load the selected model
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=VisionRunningMode.IMAGE  # Run on single images
        )
        self.landmarker = PoseLandmarker.create_from_options(options)

        self.mp_drawing = mp.solutions.drawing_utils

    def analyze_posture(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run the model
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        results = self.landmarker.detect(mp_image)

        knees_bent = False
        on_toes = False

        if results.pose_landmarks:
            landmarks = results.pose_landmarks[0]  # Get first detected person

            left_knee_angle = self.calculate_angle(landmarks[23], landmarks[25], landmarks[27])  # Left
            right_knee_angle = self.calculate_angle(landmarks[24], landmarks[26], landmarks[28])  # Right
            knees_bent = left_knee_angle < self.knee_threshold or right_knee_angle < self.knee_threshold

            left_ankle_y = landmarks[27].y
            right_ankle_y = landmarks[28].y
            on_toes = left_ankle_y < self.ankle_height_threshold and right_ankle_y < self.ankle_height_threshold

            image = self.draw_landmarks(image, landmarks)

        return {"knees_bent": knees_bent, "on_toes": on_toes}, image

    def calculate_angle(self, point1, point2, point3):
        dx1, dy1 = point1.x - point2.x, point1.y - point2.y
        dx2, dy2 = point3.x - point2.x, point3.y - point2.y
        angle = np.abs(np.arctan2(dy1, dx1) - np.arctan2(dy2, dx2))
        if angle > np.pi:
            angle = 2 * np.pi - angle
        return np.degrees(angle)

    def draw_landmarks(self, image, landmarks):
        annotated_image = image.copy()
        for landmark in landmarks:
            x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
            cv2.circle(annotated_image, (x, y), 3, (0, 255, 0), -1)
        return annotated_image

if __name__ == "__main__":
    model_path = "pose_landmark_light.task"

    analyzer = PostureChecker(model_path)

    image = cv2.imread("MeasureMates/imgs/normal.png")

    if image is None:
        print("Error: Image not found or could not be loaded.")
        exit()

    result, annotated_image = analyzer.analyze_posture(image)

    print(result)

    cv2.imshow("Pose Analysis", annotated_image)
    while True:
        key = cv2.waitKey(1) & 0xFF  
        if key == ord('q'):  # Press 'q' to close the window
            break

    cv2.destroyAllWindows()
