import os
import cv2
import numpy as np
import requests


def get_face(image_url: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    img_data = requests.get(image_url).content
    with open('modelS/images/image.jpg', 'wb') as handler:
        handler.write(img_data)
        
    
    # 1. Classifier laden
    face_cascade = cv2.CascadeClassifier("modelS/detection/haarcascade_frontalface_default.xml")

    # 2. Ordner durchlaufen, Bildpfad erstellen, Zähler erstellen und Bilder laden
    dirname = os.path.dirname(__file__)

    relevanter_Ordner = os.path.join(dirname, "images")



    for root, dirs, bilder in os.walk(relevanter_Ordner):
        for bild_name in bilder:
            bildpfad = os.path.join(root, bild_name)
            originalbild = cv2.imread(bildpfad)

            
            3.#Bildbearbeitung durchführen
            if originalbild is not None:
                # Von BGR-> In RGB und Graustufen konvertieren
                bild_rgb = cv2.cvtColor(originalbild, cv2.COLOR_BGR2RGB)
                bild_grau = cv2.cvtColor(bild_rgb, cv2.COLOR_RGB2GRAY)

                # 4. Gesichtserkennung
                faces = face_cascade.detectMultiScale(bild_grau, scaleFactor=1.3, minNeighbors=4)

                if len(faces) != 1:
                    continue 

                for (x, y, w, h) in faces:
                    
                    # Region of Interest definieren
                    roi_color = bild_rgb[y:y + h, x:x + w]

                    # Gesicht markieren
                    gesicht=cv2.rectangle(bild_rgb, (x, y), (x + w, y + h), (255, 0, 150), 4)

                    #5. Gesicht ausschneiden und in 75x100 speichern
                    gesicht_klein = cv2.resize(roi_color, (75, 100))
        
    return originalbild, gesicht, gesicht_klein
