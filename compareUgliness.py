import math
import cv2
import dlib
import matplotlib.pyplot as plt

JAW_POINTS = [0, 16]
LEFT_EYEBROW_POINTS = [17, 21]
RIGHT_EYEBROW_POINTS = [22, 26]
NOSE_POINTS = [27, 35]
LEFT_EYE_POINTS = [36, 41]
RIGHT_EYE_POINTS = [42, 47]
MOUTH_POINTS = [48, 60]
LIPS_POINTS = [61, 67]

RESIZED_HEIGHT = 200
RESIZED_WIDTH = 200


def getFaceLandmarks(image):
    # Load the detector
    detector = dlib.get_frontal_face_detector()

    # Load the predictor
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Change color space
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(imageRGB, cv2.COLOR_BGR2GRAY)

    faces = detector(imageRGB)

    for face in faces:
        x1 = face.left()
        x2 = face.right()
        y1 = face.top()
        y2 = face.bottom()

        # cv2.rectangle(img=image, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=4)

        # Look for the landmarks
        landmarks = predictor(image=gray, box=face)

        return landmarks

def getDistance(x1, y1, x2, y2):

    deltaX = x2 - x1
    deltaY = y2 - y1

    return math.sqrt(math.pow(deltaX, 2) + math.pow(deltaY, 2))

def estimateUgliness(landmarks):
    print()


def drawFeaturesPoint(landmarks, image):
    # Draw the points

    for i in range(0, 68):
        x = landmarks.part(i).x
        y = landmarks.part(i).y

        # Draw a circle
        cv2.circle(img=image, center=(x, y), radius=4, color=(0, 255, 0), thickness=-1)


def showImage(image):
    cv2.imshow("Face", mat=image)

    # Wait for a key press to exit
    cv2.waitKey(delay=0)

    # Close all windows
    cv2.destroyAllWindows()

# # Resize Image
# resized = cv2.resize(image, (RESIZED_WIDTH, RESIZED_HEIGHT))

# # Rotate image
# image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
#
# # Other ways of rotating
# rows, cols = image.shape[:2]
# deg = 45
#
# # (col/2,rows/2) is the center of rotation for the image
# # M is the coordinates of the center
# M = cv2.getRotationMatrix2D((cols/2, rows/2), deg, 1)
# image = cv2.warpAffine(image, M, (cols, rows))

# # Show image
# plt.imshow(image)
# plt.show()

# Save image
# cv2.imwrite("richardSavedTest.jpg", image)
