import math
import cv2
import dlib
import os
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
LEFT_NOSE_POINTS = [31, 33]
RIGHT_NOSE_POINTS = [34, 36]

# Eyes
LEFT_EYE_POINTS = [36, 41]
RIGHT_EYE_POINTS = [42, 47]

# Mouths
MOUTH_POINTS = [48, 60]
RIGHT_MOUTH_POINTS = [52, 53, 54, 55, 56]
LEFT_MOUTH_POINTS = [48, 49, 50, 58, 59]

# Lips
LIPS_POINTS = [61, 67]
LEFT_LIPS_POINTS = [60, 61, 67]
RIGHT_LIPS_POINTS = [63, 64, 65]

SYMMETRY_LINE_POINTS = [27, 28, 29, 30, 33, 51, 62, 66, 57, 8]

FACE_FEATURES = ["jaw",
                 "eyes",
                 "eye_brows",
                 "nose",
                 "mouth",
                 "lips"
                 ]

# Richard Reference ratio
RICHARD_RATIO_JAW = 0.84
RICHARD_RATIO_EYES = 1.239
RICHARD_RATIO_EYE_BROWS = 2.654
RICHARD_RATIO_NOSE = 1.288
RICHARD_RATIO_MOUTH = 0.817
RICHARD_RATIO_LIPS = 0.797
RICHARD_RATIO = [RICHARD_RATIO_JAW, RICHARD_RATIO_EYES, RICHARD_RATIO_EYE_BROWS,
                 RICHARD_RATIO_NOSE, RICHARD_RATIO_MOUTH, RICHARD_RATIO_LIPS]


def printCoordinate(start, end, landmarks):
    for i in range(start, end):
        x = landmarks.part(i).x
        y = landmarks.part(i).y

        print("point : ", i, "x : ", x, "y : ", y)


def getAllFileInFolder(folderPath):
    imagesPath = []

    for root, directories, files in os.walk(folderPath):
        for file in files:
            filePath = os.path.join(root, file)
            if file.endswith(".png"):
                imagesPath.append(filePath)

            if file.endswith(".jpg"):
                imagesPath.append(filePath)

            if file.endswith(".jpeg"):
                imagesPath.append(filePath)

    return imagesPath


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
    personRatioFeaturesList = getAllFaceFeaturesRatio(landmarks)
    faceFeaturesRatio = {}
    faceFeatureRatioList = []

    for featureIndex in range(0, len(FACE_FEATURES)):
        personRatio = personRatioFeaturesList[featureIndex]

        faceFeature = FACE_FEATURES[featureIndex]

        richardRatio = RICHARD_RATIO[featureIndex]

        percentageFeature = compareRatio(richardRatio, personRatio)

        faceFeaturesRatio[faceFeature] = percentageFeature

        faceFeatureRatioList.append(percentageFeature)

    beautyPercentage = averageValue(faceFeatureRatioList)
    uglinessPercentage = 100 - beautyPercentage

    return uglinessPercentage


def getUglinessFromImage(imagePath):
    name = getFileName(imagePath)

    image = cv2.imread(imagePath)

    personLandmarks = getFaceLandmarks(image)

    personUglinessPercentage = estimateUgliness(personLandmarks)

    personUglinessOn10 = percentageTo10(personUglinessPercentage)

    return personUglinessOn10


def getFileName(filePath):
    filename = os.path.basename(filePath)
    separator = '.'
    name = filename.split(separator, 1)[0]
    return name


def compareRatio(richardRatio, personRatio):
    if richardRatio > personRatio:
        return round(((100 * personRatio) / richardRatio), 2)
    else:
        return round(((100 * richardRatio) / personRatio), 2)


def drawFeaturesPoint(landmarks, image):
    # Draw the points

    for i in range(0, 68):
        x = landmarks.part(i).x
        y = landmarks.part(i).y

        # Draw a circle
        cv2.circle(img=image, center=(x, y), radius=4, color=(0, 255, 0), thickness=-1)


def getAllFaceFeaturesRatio(landmarks):
    featureRatio = []

    for feature in FACE_FEATURES:
        featureRatio.append(symmetryRatioFaceFeature(landmarks, feature))

    return featureRatio


def symmetryRatioFaceFeature(landmarks, faceFeature):
    faceFeaturesCoordinates = getAllCoordinateFaceFeature(landmarks)
    centerFaceX = getCenterFaceX(landmarks)
    ratioFeatureList = []

    if faceFeature == FACE_FEATURES[0]:
        faceCoordinateIndex = 0

    elif faceFeature == FACE_FEATURES[1]:
        faceCoordinateIndex = 1

    elif faceFeature == FACE_FEATURES[2]:
        faceCoordinateIndex = 2

    elif faceFeature == FACE_FEATURES[3]:
        faceCoordinateIndex = 3

    elif faceFeature == FACE_FEATURES[4]:
        faceCoordinateIndex = 4

    elif faceFeature == FACE_FEATURES[5]:
        faceCoordinateIndex = 5

    leftFeatureCoordinate_X = faceFeaturesCoordinates[faceCoordinateIndex][0]
    leftFeatureCoordinate_Y = faceFeaturesCoordinates[faceCoordinateIndex][1]

    rightFeatureCoordinate_X = faceFeaturesCoordinates[faceCoordinateIndex][2]
    rightFeatureCoordinate_Y = faceFeaturesCoordinates[faceCoordinateIndex][3]

    for index in range(0, len(leftFeatureCoordinate_X)):
        leftPoint_X = leftFeatureCoordinate_X[index]
        leftPoint_Y = leftFeatureCoordinate_Y[index]

        rightPoint_X = rightFeatureCoordinate_X[index]
        rightPoint_Y = rightFeatureCoordinate_Y[index]

        centerPoint_Y = centerOfPoints([leftPoint_Y, rightPoint_Y])

        leftDistance = getDistance(leftPoint_X, leftPoint_Y, centerFaceX, centerPoint_Y)
        rightDistance = getDistance(rightPoint_X, rightPoint_Y, centerFaceX, centerPoint_Y)
        ratio = leftDistance / rightDistance
        ratio = round(ratio, 3)
        ratioFeatureList.append(ratio)

    return averageValue(ratioFeatureList)


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
        index_points_end = LEFT_JAW_POINTS[1]

        for point_index in range(index_points_beginning, index_points_end):
            pointCoordinate_X = landmarks.part(point_index).x
            pointCoordinate_Y = landmarks.part(point_index).y

            leftCoordinateFeature_X.append(pointCoordinate_X)
            leftCoordinateFeature_Y.append(pointCoordinate_Y)

        index_points_beginning = RIGHT_JAW_POINTS[0]
        index_points_end = RIGHT_JAW_POINTS[1]

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

        for point_index in RIGHT_MOUTH_POINTS:
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


def averageValue(valueList):
    sumRatio = 0
    for ratio in valueList:
        sumRatio += ratio

    average = sumRatio / len(valueList)
    return round(average, 3)


def getCenterFaceX(landmarks):
    sumPoint = 0

    for pointIndex in SYMMETRY_LINE_POINTS:
        x = landmarks.part(pointIndex).x
        sumPoint += x

    return sumPoint / len(SYMMETRY_LINE_POINTS)


def getAllCoordinateFaceFeature(landmarks):
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
    COORDINATES_JAW = getCoordinateFaceFeature(FACE_FEATURES[0], landmarks)

    # Coordinates of Eyes points
    COORDINATES_EYES = getCoordinateFaceFeature(FACE_FEATURES[1], landmarks)

    # Coordinates of eye_brows points
    COORDINATES_EYE_BROWS = getCoordinateFaceFeature(FACE_FEATURES[2], landmarks)

    # Coordinates of Nose points
    COORDINATES_NOSE = getCoordinateFaceFeature(FACE_FEATURES[3], landmarks)

    # Coordinates of mouth points
    COORDINATES_MOUTH = getCoordinateFaceFeature(FACE_FEATURES[4], landmarks)

    # Coordinates of lips points
    COORDINATES_LIPS = getCoordinateFaceFeature(FACE_FEATURES[5], landmarks)

    faceFeaturesCoordinates = [COORDINATES_JAW, COORDINATES_EYES, COORDINATES_EYE_BROWS, COORDINATES_NOSE,
                               COORDINATES_MOUTH, COORDINATES_LIPS]

    return faceFeaturesCoordinates


def emptyCoordinateList(list_x, list_y):
    list_x[:] = []
    list_y[:] = []


def percentageToScale(percentage, scaleLimit):
    scale = (percentage * scaleLimit) / 100

    return round(scale)


def percentageTo10(percentage):
    return percentageToScale(percentage, 10)


def showImage(image):
    cv2.imshow("Face", mat=image)

    # Wait for a key press to exit
    cv2.waitKey(delay=0)

    # Close all windows
    cv2.destroyAllWindows()


def showImageTime(image, time):
    cv2.imshow("Picture", mat=image)

    # Wait for a key press to exit
    cv2.waitKey(delay=time*1000)

    # Close all windows
    cv2.destroyAllWindows()

# # Resize Image
# resized = cv2.resize(image, (RESIZED_WIDTH, RESIZED_HEIGHT))

# # Rotate image
# image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)


# # Other ways of rotating
# rows, cols = image.shape[:2]
# deg = 45


# # (col/2,rows/2) is the center of rotation for the image
# # M is the coordinates of the center
# M = cv2.getRotationMatrix2D((cols/2, rows/2), deg, 1)
# image = cv2.warpAffine(image, M, (cols, rows))

# # Show image
# plt.imshow(image)
# plt.show()

# Save image
# cv2.imwrite("richardSavedTest.jpg", image)
