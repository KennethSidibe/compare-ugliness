import dlib
import cv2
from compareUgliness import *

IMAGE_TEST = "armandguigma.jpg"

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Load image
    image = cv2.imread(IMAGE_TEST)

    landmarks = getFaceLandmarks(image)

    # drawFeaturesPoint(landmarks, image)

    x_reference = landmarks.part(27).x
    y_reference = landmarks.part(27).y

    x_left = landmarks.part(38).x
    y_left = landmarks.part(38).y

    x_right = landmarks.part(43).x
    y_right = landmarks.part(43).y

    # symmetryRatioFaceFeature(landmarks)

    # showImage(image)
