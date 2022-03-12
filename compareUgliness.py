import math
import cv2
import dlib
import matplotlib.pyplot as plt

# landmarks that define the human face

# JAW
JAW_POINTS = [0, 16]
LEFT_JAW_POINTS = [0, 7]
RIGHT_JAW_POINTS = [9, 16]

# Eyebrows
RIGHT_EYEBROW_POINTS = [22, 26]
LEFT_EYEBROW_POINTS = [17, 21]

# Nose
NOSE_POINTS = [27, 35]
LEFT_NOSE_POINTS = [34, 35]
RIGHT_NOSE_POINTS = [31, 32]

# Eyes
LEFT_EYE_POINTS = [36, 41]
RIGHT_EYE_POINTS = [42, 47]

# Mouths
MOUTH_POINTS = [48, 60]
LEFT_MOUTH_POINTS = [52, 53, 54, 55, 56]
RIGHT_MOUTH_POINTS = [48, 49, 50, 58, 59]

# Lips
LIPS_POINTS = [61, 67]
LEFT_LIPS_POINTS = [60, 61, 67]
RIGHT_LIPS_POINTS = [63, 64, 65]


SYMMETRY_LINE_POINTS = [27, 28, 29, 30, 33, 51, 62, 66, 57, 8]
FACE_FEATURES = ["eyes",
                 "eye_brows",
                 "nose",
                 "lips",
                 "mouth",
                 "jaw"
                 ]

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


def symmetryRatioFaceFeature(faceFeature):
    #     TODO
    print()


def centerOfPoints(coordinates):
    sumOfCoordinates = 0

    for number in coordinates:
        sumOfCoordinates += number

    return sumOfCoordinates / len(coordinates)

def getCoordinateFaceFeature(feature, landmarks):
    leftCoordinateFeature_X = []
    leftCoordinateFeature_Y = []

    rightCoordinateFeature_X = []
    rightCoordinateFeature_Y = []

    coordinates = []

    if feature == "jaw":
        index_points_beginning = LEFT_JAW_POINTS[0]
        index_points_end = RIGHT_JAW_POINTS[1]

        for point_index in range(index_points_beginning, index_points_end):
            pointCoordinate_X = landmarks.part(point_index).x
            pointCoordinate_Y = landmarks.part(point_index).y

            leftCoordinateFeature_X.append(pointCoordinate_X)
            leftCoordinateFeature_Y.append(pointCoordinate_Y)

        index_points_beginning = RIGHT_EYE_POINTS[0]
        index_points_end = RIGHT_EYE_POINTS[1]

        for point_index in range(index_points_beginning, index_points_end):
            pointCoordinate_X = landmarks.part(point_index).x
            pointCoordinate_Y = landmarks.part(point_index).y

            rightCoordinateFeature_X.append(pointCoordinate_X)
            rightCoordinateFeature_Y.append(pointCoordinate_Y)

        coordinates.append(leftCoordinateFeature_X)
        coordinates.append(leftCoordinateFeature_Y)

        coordinates.append(rightCoordinateFeature_X)
        coordinates.append(rightCoordinateFeature_Y)

        return coordinates

    if feature == "eyes":
        index_points_beginning = LEFT_EYE_POINTS[0]
        index_points_end = LEFT_EYE_POINTS[1]

        for point_index in range(index_points_beginning, index_points_end):
            pointCoordinate_X = landmarks.part(point_index).x
            pointCoordinate_Y = landmarks.part(point_index).y

            leftCoordinateFeature_X.append(pointCoordinate_X)
            leftCoordinateFeature_Y.append(pointCoordinate_Y)

        index_points_beginning = RIGHT_EYE_POINTS[0]
        index_points_end = RIGHT_EYE_POINTS[1]

        for point_index in range(index_points_beginning, index_points_end):
            pointCoordinate_X = landmarks.part(point_index).x
            pointCoordinate_Y = landmarks.part(point_index).y

            rightCoordinateFeature_X.append(pointCoordinate_X)
            rightCoordinateFeature_Y.append(pointCoordinate_Y)

        coordinates.append(leftCoordinateFeature_X)
        coordinates.append(leftCoordinateFeature_Y)

        coordinates.append(rightCoordinateFeature_X)
        coordinates.append(rightCoordinateFeature_Y)

        return coordinates

    if feature == "eye_brows":

        index_points_beginning = LEFT_EYEBROW_POINTS[0]
        index_points_end = LEFT_EYEBROW_POINTS[1]

        for point_index in range(index_points_beginning, index_points_end):
            pointCoordinate_X = landmarks.part(point_index).x
            pointCoordinate_Y = landmarks.part(point_index).y

            leftCoordinateFeature_X.append(pointCoordinate_X)
            leftCoordinateFeature_Y.append(pointCoordinate_Y)

        index_points_beginning = RIGHT_EYEBROW_POINTS[0]
        index_points_end = RIGHT_EYEBROW_POINTS[1]

        for point_index in range(index_points_beginning, index_points_end):
            pointCoordinate_X = landmarks.part(point_index).x
            pointCoordinate_Y = landmarks.part(point_index).y

            rightCoordinateFeature_X.append(pointCoordinate_X)
            rightCoordinateFeature_Y.append(pointCoordinate_Y)

        coordinates.append(leftCoordinateFeature_X)
        coordinates.append(leftCoordinateFeature_Y)

        coordinates.append(rightCoordinateFeature_X)
        coordinates.append(rightCoordinateFeature_Y)

        return coordinates

    if feature == "nose":

        index_points_beginning = LEFT_NOSE_POINTS[0]
        index_points_end = LEFT_NOSE_POINTS[1]

        for point_index in range(index_points_beginning, index_points_end):
            pointCoordinate_X = landmarks.part(point_index).x
            pointCoordinate_Y = landmarks.part(point_index).y

            leftCoordinateFeature_X.append(pointCoordinate_X)
            leftCoordinateFeature_Y.append(pointCoordinate_Y)

        index_points_beginning = RIGHT_NOSE_POINTS[0]
        index_points_end = RIGHT_NOSE_POINTS[1]

        for point_index in range(index_points_beginning, index_points_end):
            pointCoordinate_X = landmarks.part(point_index).x
            pointCoordinate_Y = landmarks.part(point_index).y

            rightCoordinateFeature_X.append(pointCoordinate_X)
            rightCoordinateFeature_Y.append(pointCoordinate_Y)

        coordinates.append(leftCoordinateFeature_X)
        coordinates.append(leftCoordinateFeature_Y)

        coordinates.append(rightCoordinateFeature_X)
        coordinates.append(rightCoordinateFeature_Y)

        return coordinates

    if feature == "mouth":

        for point_index in LEFT_MOUTH_POINTS:
            pointCoordinate_X = landmarks.part(point_index).x
            pointCoordinate_Y = landmarks.part(point_index).y

            leftCoordinateFeature_X.append(pointCoordinate_X)
            leftCoordinateFeature_Y.append(pointCoordinate_Y)

        for point_index in LEFT_MOUTH_POINTS:
            pointCoordinate_X = landmarks.part(point_index).x
            pointCoordinate_Y = landmarks.part(point_index).y

            rightCoordinateFeature_X.append(pointCoordinate_X)
            rightCoordinateFeature_Y.append(pointCoordinate_Y)

        coordinates.append(leftCoordinateFeature_X)
        coordinates.append(leftCoordinateFeature_Y)

        coordinates.append(rightCoordinateFeature_X)
        coordinates.append(rightCoordinateFeature_Y)

        return coordinates

    if feature == "lips":

        for point_index in LEFT_LIPS_POINTS:
            pointCoordinate_X = landmarks.part(point_index).x
            pointCoordinate_Y = landmarks.part(point_index).y

            leftCoordinateFeature_X.append(pointCoordinate_X)
            leftCoordinateFeature_Y.append(pointCoordinate_Y)

        for point_index in RIGHT_LIPS_POINTS:
            pointCoordinate_X = landmarks.part(point_index).x
            pointCoordinate_Y = landmarks.part(point_index).y

            rightCoordinateFeature_X.append(pointCoordinate_X)
            rightCoordinateFeature_Y.append(pointCoordinate_Y)

        coordinates.append(leftCoordinateFeature_X)
        coordinates.append(leftCoordinateFeature_Y)

        coordinates.append(rightCoordinateFeature_X)
        coordinates.append(rightCoordinateFeature_Y)

        return coordinates

def emptyCoordinateList(list_x, list_y):
    list_x [:] = []
    list_y [:] = []

def getCoordinateFaceFeature(landmarks):
    leftCoordinateFeature_X = []
    leftCoordinateFeature_Y = []

    rightCoordinateFeature_X = []
    rightCoordinateFeature_Y = []

    COORDINATES_EYES = []
    COORDINATES_EYE_BROWS = []
    COORDINATES_NOSE = []
    COORDINATES_MOUTH = []
    COORDINATES_LIPS = []
    COORDINATES_JAW = []

    # coordinates of Jaws points

    index_points_beginning = LEFT_JAW_POINTS[0]
    index_points_end = RIGHT_JAW_POINTS[1]

    # Left jaw points
    for point_index in range(index_points_beginning, index_points_end):
        pointCoordinate_X = landmarks.part(point_index).x
        pointCoordinate_Y = landmarks.part(point_index).y

        leftCoordinateFeature_X.append(pointCoordinate_X)
        leftCoordinateFeature_Y.append(pointCoordinate_Y)

    # Right Jaw Points
    index_points_beginning = RIGHT_JAW_POINTS[0]
    index_points_end = RIGHT_JAW_POINTS[1]

    for point_index in range(index_points_beginning, index_points_end):
        pointCoordinate_X = landmarks.part(point_index).x
        pointCoordinate_Y = landmarks.part(point_index).y

        rightCoordinateFeature_X.append(pointCoordinate_X)
        rightCoordinateFeature_Y.append(pointCoordinate_Y)

    COORDINATES_JAW.append(leftCoordinateFeature_X)
    COORDINATES_JAW.append(leftCoordinateFeature_Y)

    COORDINATES_JAW.append(rightCoordinateFeature_X)
    COORDINATES_JAW.append(rightCoordinateFeature_Y)

    # empty the list
    rightCoordinateFeature_X[:] = []
    rightCoordinateFeature_Y[:] = []

    leftCoordinateFeature_X[:] = []
    leftCoordinateFeature_Y[:] = []

    # Coordinates of Eyes points

    # Left eye Points
    index_points_beginning = LEFT_EYE_POINTS[0]
    index_points_end = LEFT_EYE_POINTS[1]

    for point_index in range(index_points_beginning, index_points_end):
        pointCoordinate_X = landmarks.part(point_index).x
        pointCoordinate_Y = landmarks.part(point_index).y

        leftCoordinateFeature_X.append(pointCoordinate_X)
        leftCoordinateFeature_Y.append(pointCoordinate_Y)

    index_points_beginning = RIGHT_EYE_POINTS[0]
    index_points_end = RIGHT_EYE_POINTS[1]

    # Right eye Points
    for point_index in range(index_points_beginning, index_points_end):
        pointCoordinate_X = landmarks.part(point_index).x
        pointCoordinate_Y = landmarks.part(point_index).y

        rightCoordinateFeature_X.append(pointCoordinate_X)
        rightCoordinateFeature_Y.append(pointCoordinate_Y)

    COORDINATES_EYES.append(leftCoordinateFeature_X)
    COORDINATES_EYES.append(leftCoordinateFeature_Y)

    COORDINATES_EYES.append(rightCoordinateFeature_X)
    COORDINATES_EYES.append(rightCoordinateFeature_Y)


    # Coordinates of eye_brows points

    # Right eyebrow points
    index_points_beginning = LEFT_EYEBROW_POINTS[0]
    index_points_end = LEFT_EYEBROW_POINTS[1]

    for point_index in range(index_points_beginning, index_points_end):
        pointCoordinate_X = landmarks.part(point_index).x
        pointCoordinate_Y = landmarks.part(point_index).y

        leftCoordinateFeature_X.append(pointCoordinate_X)
        leftCoordinateFeature_Y.append(pointCoordinate_Y)

    index_points_beginning = RIGHT_EYEBROW_POINTS[0]
    index_points_end = RIGHT_EYEBROW_POINTS[1]

    # Left eyebrow points

    for point_index in range(index_points_beginning, index_points_end):
        pointCoordinate_X = landmarks.part(point_index).x
        pointCoordinate_Y = landmarks.part(point_index).y

        rightCoordinateFeature_X.append(pointCoordinate_X)
        rightCoordinateFeature_Y.append(pointCoordinate_Y)

    COORDINATES_EYE_BROWS.append(leftCoordinateFeature_X)
    COORDINATES_EYE_BROWS.append(leftCoordinateFeature_Y)

    COORDINATES_EYE_BROWS.append(rightCoordinateFeature_X)
    COORDINATES_EYE_BROWS.append(rightCoordinateFeature_Y)

    # Coordinates of Nose points

    # Right nose points

    index_points_beginning = LEFT_NOSE_POINTS[0]
    index_points_end = LEFT_NOSE_POINTS[1]

    for point_index in range(index_points_beginning, index_points_end):
        pointCoordinate_X = landmarks.part(point_index).x
        pointCoordinate_Y = landmarks.part(point_index).y

        leftCoordinateFeature_X.append(pointCoordinate_X)
        leftCoordinateFeature_Y.append(pointCoordinate_Y)

    index_points_beginning = RIGHT_NOSE_POINTS[0]
    index_points_end = RIGHT_NOSE_POINTS[1]

    # left nose points

    for point_index in range(index_points_beginning, index_points_end):
        pointCoordinate_X = landmarks.part(point_index).x
        pointCoordinate_Y = landmarks.part(point_index).y

        rightCoordinateFeature_X.append(pointCoordinate_X)
        rightCoordinateFeature_Y.append(pointCoordinate_Y)

    COORDINATES_NOSE.append(leftCoordinateFeature_X)
    COORDINATES_NOSE.append(leftCoordinateFeature_Y)

    COORDINATES_NOSE.append(rightCoordinateFeature_X)
    COORDINATES_NOSE.append(rightCoordinateFeature_Y)

    # Coordinates of mouth points

    # left mouth points
    for point_index in LEFT_MOUTH_POINTS:
        pointCoordinate_X = landmarks.part(point_index).x
        pointCoordinate_Y = landmarks.part(point_index).y

        leftCoordinateFeature_X.append(pointCoordinate_X)
        leftCoordinateFeature_Y.append(pointCoordinate_Y)

    # Right mouth points
    for point_index in LEFT_MOUTH_POINTS:
        pointCoordinate_X = landmarks.part(point_index).x
        pointCoordinate_Y = landmarks.part(point_index).y

        rightCoordinateFeature_X.append(pointCoordinate_X)
        rightCoordinateFeature_Y.append(pointCoordinate_Y)

    COORDINATES_MOUTH.append(leftCoordinateFeature_X)
    COORDINATES_MOUTH.append(leftCoordinateFeature_Y)

    COORDINATES_MOUTH.append(rightCoordinateFeature_X)
    COORDINATES_MOUTH.append(rightCoordinateFeature_Y)


    # Coordinates of lips points

    # left lips points
    for point_index in LEFT_LIPS_POINTS:
        pointCoordinate_X = landmarks.part(point_index).x
        pointCoordinate_Y = landmarks.part(point_index).y

        leftCoordinateFeature_X.append(pointCoordinate_X)
        leftCoordinateFeature_Y.append(pointCoordinate_Y)

    # right lips points
    for point_index in RIGHT_LIPS_POINTS:
        pointCoordinate_X = landmarks.part(point_index).x
        pointCoordinate_Y = landmarks.part(point_index).y

        rightCoordinateFeature_X.append(pointCoordinate_X)
        rightCoordinateFeature_Y.append(pointCoordinate_Y)

    COORDINATES_LIPS.append(leftCoordinateFeature_X)
    COORDINATES_LIPS.append(leftCoordinateFeature_Y)

    COORDINATES_LIPS.append(rightCoordinateFeature_X)
    COORDINATES_LIPS.append(rightCoordinateFeature_Y)



def symmetryRatioFaceFeature(landmarks):
    coordinateSymmetryLine_X = []
    coordinateSymmetryLine_Y = []

    for i in SYMMETRY_LINE_POINTS:
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        coordinateSymmetryLine_X.append(x)
        coordinateSymmetryLine_Y.append(y)

    SYMMETRY_X_COORDINATE = centerOfPoints(coordinateSymmetryLine_X)

    for feature in FACE_FEATURES:
        leftEyeCoordinate_X = []
        leftEyeCoordinate_y = []

#         todo



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
