import cv2

class LengthMeasurer:
    def __init__(self, min_box_area=2000, poster_length=200):
        self.min_box_area = min_box_area
        self.poster_length = poster_length

    def calc_length(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_contours = [contour for contour in contours if cv2.contourArea(contour) >= self.min_box_area]

        cv2.drawContours(image, valid_contours, -1, (0, 255, 0), 2)  # Green color, thickness=2

        number_of_boxes = len(valid_contours)
        print(f"Number of boxes: {number_of_boxes}")
        child_length = self.poster_length - number_of_boxes 

        return child_length, image


if __name__ == '__main__':
    image_path = 'imgs/ruler2.png'
    image = cv2.imread(image_path)  

    box_counter = LengthMeasurer()
    number_of_boxes, annotated_image = box_counter.calc_length(image)

    print(f"Number of boxes: {number_of_boxes}")

    # Show the image with the detected boxes
    cv2.imshow("Detected Boxes", annotated_image)
    while True:
        key = cv2.waitKey(1) & 0xFF  # Capture key press

        if key == ord('q'):  # Press 'q' to close the window
            break    
    cv2.destroyAllWindows()
