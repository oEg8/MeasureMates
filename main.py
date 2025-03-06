import os
import requests
import cv2
from length_measurer import LengthMeasurer
from posture_checker import PostureChecker

MODEL_CHOICES = {
    "light": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/lite/float16/1/pose_landmarker_lite.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/heavy/float16/1/pose_landmarker_heavy.task"
}

MODEL_PATH = "pose_landmark_light.task"  # for the light version
# MODEL_PATH = "pose_landmark_heavy.task"  # for the heavy version (uncomment this line)

def download_model(model_path, model_url):
    """Downloads the model if it's not already present."""
    if not os.path.exists(model_path):
        print(f"Downloading model: {model_path} ...")
        response = requests.get(model_url, stream=True)
        if response.status_code == 200:
            with open(model_path, "wb") as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            print("Model downloaded successfully!")
        else:
            print("Error downloading the model. Please check the URL and try again.")
            exit()
    else:
        print("Model already exists. Skipping download.")

def main():
    download_model(MODEL_PATH, MODEL_CHOICES["light"])  # Change "light" to "heavy" if needed

    image_path = 'imgs/normal.png'
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Image not found or could not be loaded.")
        exit()

    # Pass model path to both classes
    box_counter = LengthMeasurer(MODEL_PATH)
    child_length, image = box_counter.calc_length(image)

    ps = PostureChecker(MODEL_PATH)
    result, image = ps.analyze_posture(image)

    print(f"The client is {child_length}cm tall")
    print(f"The client is standing with knees bent: {result['knees_bent']}")
    print(f"The client is standing on their toes: {result['on_toes']}")

    cv2.imshow("Pose Analysis", image)
    while True:
        key = cv2.waitKey(1) & 0xFF  
        if key == ord('q'):  # Press 'q' to close the window
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
