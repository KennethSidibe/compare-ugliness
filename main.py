import dlib
import cv2
import time
from compareUgliness import *

IMAGE_TEST = "armandguigma.jpg"
RICHARD_PIC = "Richardphoto.png"

FRIENDS_IMAGE_FOLDER_PATH = rf"C:\Users\Kenneth Sidibe\PycharmProjects\compareUgliness\MGTC Pic"

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    imagesPath = getAllFileInFolder(FRIENDS_IMAGE_FOLDER_PATH)

    # Load image
    # image = cv2.imread(IMAGE_TEST)

    # landmarks = getFaceLandmarks(image)

    # drawFeaturesPoint(landmarks, image)

    # percentage = estimateUgliness(landmarks)
    uglinessRate = []

    print("BIENVENUE sur le calculateur de vilainété !")
    print()
    time.sleep(5)
    print("J'ai deja les images de tout le monde donc on peut commencer a se comparer")

    time.sleep(5)
    print("Voici l'homme le plus beau le plus frais le plus choco sur lequel je me baserai pour vous evaluer : ")
    richardImage = cv2.imread(RICHARD_PIC)
    showImageTime(richardImage, 5)
    print()

    print("Bon j'espere que vous avez pu contempler le beau passons maintenant au test")
    print()
    time.sleep(5)

    print("Voici la premiere personnes que nous allons estimer la beaute ")
    name = getFileName(imagesPath[0])
    image = cv2.imread(imagesPath[0])
    showImageTime(image, 5)
    print("Voici : ", name, "cherchons sur une echelle de 1 a 10 combien il est vilain")
    print()
    print("Loading vilainete...")
    personUgliness = getUglinessFromImage(imagesPath[0])
    uglinessRate.append(personUgliness)
    print("Alors ", name, " est ", personUgliness, "fois vilain sur une echelle de 1 a 10")
    print()
    time.sleep(5)

    print("Voici la deuxieme personne que nous allons estimer la beaute ")
    image = cv2.imread(imagesPath[1])
    name = getFileName(imagesPath[1])
    showImageTime(image, 5)
    print("Voici : ", name, "cherchons sur une echelle de 1 a 10 combien il est vilain")
    print()
    print("Loading vilainete...")
    personUgliness = getUglinessFromImage(imagesPath[1])
    uglinessRate.append(personUgliness)
    print("Alors ", name, " est ", personUgliness, "fois vilain sur une echelle de 1 a 10")
    print()
    time.sleep(5)

    print("Voici la troisieme personne que nous allons estimer la beaute ")
    image = cv2.imread(imagesPath[2])
    name = getFileName(imagesPath[2])
    showImageTime(image, 5)
    print("Voici : ", name, "cherchons sur une echelle de 1 a 10 combien il est vilain")
    print()
    print("Loading vilainete...")
    personUgliness = getUglinessFromImage(imagesPath[2])
    uglinessRate.append(personUgliness)
    print("Alors ", name, " est ", personUgliness, "fois vilain sur une echelle de 1 a 10")
    print()
    time.sleep(5)

    print("Voici la quatrieme personne que nous allons estimer la beaute ")
    image = cv2.imread(imagesPath[3])
    name = getFileName(imagesPath[3])
    showImageTime(image, 5)
    print("Voici : ", name, "cherchons sur une echelle de 1 a 10 combien il est vilain")
    print()
    print("Loading vilainete...")
    personUgliness = getUglinessFromImage(imagesPath[3])
    uglinessRate.append(personUgliness)
    print("Alors ", name, " est ", personUgliness, "fois vilain sur une echelle de 1 a 10")
    print()
    time.sleep(5)

    print("Voici la cinquieme personne que nous allons estimer la beaute ")
    image = cv2.imread(imagesPath[4])
    name = getFileName(imagesPath[4])
    showImageTime(image, 5)
    print("Voici : ", name, "cherchons sur une echelle de 1 a 10 combien il est vilain")
    print()
    print("Loading vilainete...")
    personUgliness = getUglinessFromImage(imagesPath[4])
    uglinessRate.append(personUgliness)
    print("Alors ", name, " est ", personUgliness, "fois vilain sur une echelle de 1 a 10")
    print()
    time.sleep(5)

    print("Voici la sixieme personne que nous allons estimer la beaute ")
    image = cv2.imread(imagesPath[5])
    name = getFileName(imagesPath[5])
    showImageTime(image, 5)
    print("Voici : ", name, "cherchons sur une echelle de 1 a 10 combien il est vilain")
    print()
    print("Loading vilainete...")
    personUgliness = getUglinessFromImage(imagesPath[5])
    uglinessRate.append(personUgliness)
    print("Alors ", name, " est ", personUgliness, "fois vilain sur une echelle de 1 a 10")
    print()
    time.sleep(5)

    uglinessRate.sort()

    print("voici le plus vilain maintenant :")
    time.sleep(1)
    print(".")
    time.sleep(1)
    print("..")
    time.sleep(1)
    print("...")

    showImageTime(cv2.imread(imagesPath[5]), 5)

