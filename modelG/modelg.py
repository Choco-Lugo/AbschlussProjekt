import os
import cv2
import numpy as np
import random



def verarbeite_gesicht():
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # 2. Ordner durchlaufen, Bildpfad erstellen, Zähler erstellen und Bilder laden
    relevanter_Ordner = "LFW Daten/lfw-deepfunneled/lfw-deepfunneled"

    gespeicherte_Dateinamen=[]
    gespeicherte_Gesichter=[]
    anzahl_bilder_mit_einem_gesicht = 0

    for root, dirs, bilder in os.walk(relevanter_Ordner):
        for bild_name in bilder:
            bildpfad = os.path.join(root, bild_name)
            originalbild = cv2.imread(bildpfad)
            
            #Bildbearbeitung durchführen
            if originalbild is not None:
                # In RGB und Graustufen konvertieren
                bild_rgb = cv2.cvtColor(originalbild, cv2.COLOR_BGR2RGB)
                bild_grau = cv2.cvtColor(bild_rgb, cv2.COLOR_RGB2GRAY)

                # 4. Gesichtserkennung
                faces = face_cascade.detectMultiScale(bild_grau, scaleFactor=1.1, minNeighbors=5)

                if len(faces) != 1:
                    continue 
                anzahl_bilder_mit_einem_gesicht += 1

                for (x, y, w, h) in faces:
                    # Gesicht markieren
                    gesicht=cv2.rectangle(bild_rgb, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    
                    # Region of Interest definieren
                    roi_gray = bild_grau[y:y + h, x:x + w]
                    roi_color = bild_rgb[y:y + h, x:x + w]
                    
                    #5. Gesicht ausschneiden und in 75x100 speichern
                    #Gesicht in 75x100 ausschneiden und speichern
                    gesicht_klein = cv2.resize(roi_color, (75, 100))
                        
                    # 6. gespeicherte Dateinamen und gespeicherte Gesichter in zwei verschiedenen Listen speichern
                    gespeicherte_Dateinamen.append(bild_name)
                    gespeicherte_Gesichter.append(gesicht_klein)

    #if anzahl_bilder_mit_einem_gesicht == 0:
        #raise ValueError("Keine Bilder mit genau einem Gesicht gefunden.")

    #7.Gespeicherte Dateinamen und gespeicherte Gesichter in numpy arrays umwandeln
    gespeicherte_Dateinamen_np_array= np.array(gespeicherte_Dateinamen)
    gespeicherte_Gesichter_np_array= np.array(gespeicherte_Gesichter)
            
    #8.Arrays in .npy speichern
    np.save("gespeicherte_Dateinamen_LFW.npy", gespeicherte_Dateinamen_np_array)
    np.save("gespeicherte_Gesichter_LFW.npy", gespeicherte_Gesichter_np_array)

    zufallsbilder = random.sample(gespeicherte_Gesichter, min(5, len(gespeicherte_Gesichter)))

    return originalbild, bild_grau, bild_rgb, gesicht_klein, zufallsbilder, 13233, anzahl_bilder_mit_einem_gesicht, (anzahl_bilder_mit_einem_gesicht/13233)*100  