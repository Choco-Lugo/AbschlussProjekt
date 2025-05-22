import streamlit as st
import os
import cv2
import gc
import random
import matplotlib.pyplot as plt
import numpy as np
import pickle
from PIL import Image
from sklearn.decomposition import PCA
from pathlib import Path
from typing import List, Tuple

import start, camera
from modelV import model
from modelG import modelg
from modelS import face_recognition_page

st.set_page_config(layout="wide") 
model_path = "modelV/face_model.pth"

def load_or_cache_images(
        dataset_path: str,
        pickle_dir: str = "cached_images",
        block_size: int = 50000,
        target_size: Tuple[int, int] = (100, 100),
) -> List[np.ndarray]:
    """
    Lädt ALLE Bilder (jpg/jpeg/png) aus 'dataset_path'.
    - Existieren bereits Pickle-Blöcke in 'pickle_dir', werden sie direkt eingelesen.
    - Andernfalls werden die Bilder auf 'target_size' verkleinert, in Blöcken von
      'block_size' Elementen gespeichert und anschließend wieder als eine Liste
      numpy-Arrays zurückgegeben.
      """
    
    Path(pickle_dir).mkdir(parents=True, exist_ok=True)

    # ───────────────────── 1) versuchen, Cache zu laden ──────────────────────────
    cached = sorted(Path(pickle_dir).glob("images_block_*.pkl"))
    if cached:
        print(f"📦  Lade {len(cached)} Pickle-Blöcke …")
        all_images = []
        for pkl in cached:
            with pkl.open("rb") as f:
                all_images.extend(pickle.load(f))
        return all_images

    # ───────────────────── 2) neu aufbauen, wenn kein Cache ──────────────────────
    print("📸  Kein Cache gefunden – Bilder einlesen und verkleinern …")
    images, blk_idx, counter = [], 0, 0

    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = Path(root) / file
                try:
                    img = Image.open(img_path).convert("RGB").resize(target_size)
                    images.append(np.asarray(img, dtype=np.uint8))
                    counter += 1
                except Exception as e:
                    print(f"⚠️  {img_path}: {e}")

                # Block voll? → Pickle schreiben
                if counter % block_size == 0:
                    fname = Path(pickle_dir) / f"images_block_{blk_idx:03d}.pkl"
                    with fname.open("wb") as f:
                        pickle.dump(images, f)
                    print(f"💾  Block {blk_idx:03d} gespeichert  ({len(images)} Bilder)")
                    images.clear(); blk_idx += 1; gc.collect()

    # verbliebene Rest-Bilder speichern
    if images:
        fname = Path(pickle_dir) / f"images_block_{blk_idx:03d}.pkl"
        with fname.open("wb") as f:
            pickle.dump(images, f)
        print(f"💾  Block {blk_idx:03d} gespeichert  ({len(images)} Bilder)")

    # jetzt (einmalig) alle Blöcke einlesen und zurückgeben
    return load_or_cache_images(dataset_path, pickle_dir, block_size, target_size)

def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("styles.css")

if "page" not in st.session_state:
    st.session_state.page = "Herausforderung" 

st.sidebar.markdown(
    '<h2 style="color: white;">Navigation</h2>',
    unsafe_allow_html=True
)

if st.sidebar.button("🧗 Herausforderung"):
    st.session_state.page = "Herausforderung"
if st.sidebar.button("📊 Face Detection"):
    st.session_state.page = "Face Detection"
#if st.sidebar.button("📊 Analyse Gökhan"):
    #st.session_state.page = "Analyse Gökhan"
if st.sidebar.button("📊 Face Recognition"):
    st.session_state.page = "Face Recognition"
if st.sidebar.button("📷 Kamera"):
    st.session_state.page = "Kamera"

# Inhalt je nach gewählter Seite
st.title(f"{st.session_state.page}")

if st.session_state.page == "Herausforderung":
    start.app()

elif st.session_state.page == "Face Detection":
    col1, col2 = st.columns(2)

    with col1:
        st.write("Hier findet die Datenanalyse statt...")

        dataset_path = "modelV/VGG-Face2/data/vggface2_train/train"

        st.subheader("📸 Bild aus dem VGG-Face2 Datensatz:")
        with st.expander("📄 Erklärung zur Funktion `load_all_image_paths()`"):
            st.markdown("""
            ### Funktion: `load_all_image_paths(dataset_path)`

            Diese Funktion durchsucht rekursiv ein Dataset-Verzeichnis nach Bilddateien und gibt eine Liste aller gefundenen Pfade zurück.

            - Sie erwartet, dass `dataset_path` ein Ordner mit Unterordnern ist (z. B. für Personen).
            - Sie filtert nur Bilddateien mit `.jpg`, `.jpeg` oder `.png`.
            - Alle gültigen Pfade werden in einer Liste gespeichert.

            ✅ Durch `@st.cache_data` wird das Ergebnis für denselben Ordner zwischengespeichert, um Ladezeiten zu minimieren.
            """)
        
        folder_id = st.secrets["drive_ids"]["image_folder"]
        all_images = model.load_all_images_recursive(folder_id)

        st.success(f"✅ {len(all_images)} Bilder erfolgreich geladen.")

        if not os.path.exists("face1.jpg"):
            st.warning("❗ Bitte zuerst ein Bild unter 📷 Kamera aufnehmen und speichern.")
        else:
            img = cv2.imread("face1.jpg")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, caption="Gesicht", width=200)

    with col2:
        st.subheader("🛠️ Herausforderungen und Lösungen")

        st.markdown("""
        - **Bild-Upload optimiert**: Der Upload großer Bildmengen war anfangs sehr langsam.  
        Durch den Einsatz von `@st.cache_data` konnte die Ladezeit erheblich reduziert werden.

        - **Modelltraining ausgelagert**: Das Training des Modells war sehr zeitintensiv und für eine Live-Präsentation ungeeignet.  
        Daher wurde das trainierte Modell **einmalig gespeichert** und kann bei Bedarf geladen werden  
        (**Train → Save → Load**).

        - **Speichermanagement verbessert**: Beim Laden und Verarbeiten vieler Bilder kam es zu **RAM-Engpässen**,  
        insbesondere bei großen Batch-Größen oder beim vollständigen Vorladen des Datensatzes.  
        Die Lösung war eine **limitierte Verarbeitung pro Durchlauf** (z. B. `max_images`) sowie gezieltes Freigeben nicht mehr benötigter Ressourcen.

        - **Bildgrößen vereinheitlicht**: Einige Bilder hatten sehr hohe Auflösungen, was zu übermäßigem Speicherverbrauch führte.  
        Durch das **systematische Verkleinern auf einheitliche Größen** (z. B. 64×64 oder 150×200 Pixel) konnten sowohl  
        die Speicherbelastung als auch die Ladezeiten optimiert werden.
        """)

    # Modell laden oder trainieren
    with st.expander("📄 Model SimpleFaceCNN`"):
        st.markdown("""
        ### Funktion: `Model CNN Training`
                    
        Modellarchitektur:

        Einfaches CNN für Gesichtserkennung: Es wird ein Modell namens SimpleFaceCNN verwendet. Es handeld sich um ein Convolutional Neural Network, das für die Verarbeitung von Gesichtern konzipiert ist.

        Classifier und Ausgabeschicht: Das Modell hat eine Klassifikationsschicht (model.classifier), zu der eine neue Ausgabeschicht mit nn.Linear(128, len(os.listdir(root_dir))) hinzugefügt wird. Diese Schicht hat 128 Eingabeknoten (die wahrscheinlich die Anzahl der Merkmale des CNNs nach den Convolutional Layers darstellen) und eine Ausgabeschicht, die der Anzahl der verschiedenen Klassen entspricht (basierend auf der Anzahl der Unterordner in root_dir, was wahrscheinlich für die Anzahl der verschiedenen Gesichter oder Identitäten steht).

        **Erklärung**:
                                
        1. Es wird überprüft, ob ein Modell bereits auf der Festplatte gespeichert ist (durch Überprüfung des `model_path`).
        2. **Wenn das Modell bereits existiert**:
            - ✅ Das Modell wird mit `model.load_model()` geladen und in der Variablen `cnn` gespeichert.
        3. **Wenn das Modell nicht existiert**:
            - ❌ Wird das Modell mit `model.train_model()` neu trainiert, gespeichert und anschließend geladen.
        4. Während des Ladevorgangs oder Trainings sorgt `st.spinner(...)` für eine visuelle Ladeanzeige in der Streamlit-App, damit der Nutzer den Fortschritt sehen kann.
        5. **Zusätzliche Statusanzeigen**: Streamlit zeigt mit `st.info()`, `st.warning()` und `st.success()` passende Nachrichten, um den Status des Modells zu kommunizieren.
        """)

    if os.path.exists(model_path):
        st.info("📂 Vortrainiertes Modell gefunden. Wird geladen...")
        with st.spinner("Modell wird geladen..."):
            cnn = model.load_model(model_path)
    else:
        st.warning("⚠️ Kein gespeichertes Modell gefunden. Training wird gestartet...")
        with st.spinner("Modell wird trainiert..."):
            cnn = model.train_model()
            model.save_model(cnn, model_path)
            st.success("✅ Modell wurde gespeichert.")

    st.info("🔄 Zähle Gesichter mit Cascade-Modell...")
    sample_images = random.sample(all_images, min(5, len(all_images)))
    cols = st.columns(5)

    for i, img_path in enumerate(sample_images):
        num_faces = model.count_faces_in_image(img_path)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (200, 200))  
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        with cols[i]:
            st.image(img_rgb, use_column_width=False)
            st.markdown(
                f"<div style='text-align:center; color:white; font-weight:bold;'>{num_faces} Gesicht(er)</div>",
                unsafe_allow_html=True)
            
    st.subheader("📊 Modellbewertung & Visualisierung")  

    with st.expander("📄 FaceDataset"):
        st.markdown("""
        **Pickle**:  
        Um das Laden der Bilddaten zu beschleunigen und die Effizienz des Trainingsprozesses zu verbessern, wird das Dataset mithilfe von **Pickle im Client gespeichert**. Falls ein bereits gespeichertes Dataset existiert, wird es geladen, anstatt es erneut von der Festplatte zu lesen. Dies spart Zeit und Ressourcen während des Trainings.

        **Dataset**:  
        Das `FaceDataset`-Objekt ist dafür verantwortlich, Bilddaten aus einem angegebenen Verzeichnis zu laden und für das Modell vorzubereiten. Es durchsucht die Ordnerstruktur, extrahiert Bilder und ordnet jedem Bild ein Label zu, das die Identität der dargestellten Person repräsentiert. Die Bilder werden in ein geeignetes Format (Tensor) umgewandelt und sind somit für das Training mit einem CNN optimiert.
        """)


    file_id   = st.secrets["drive_ids"]["dataset_file"]

    cache_path = "AbschluProjekt/cached_dataset.pkl"
    st.info("🌐 Lade gecachtes Dataset von Google Drive...")
    model.download_from_gdrive(file_id, cache_path)
    dataset = model.load_dataset(cache_path)
    st.success("✅ Dataset erfolgreich geladen.")
    
    #accuracy, cm_fig, tsne_fig, dist_fig = model.evaluate_model(cnn, dataset)   
    #st.success(f"✅ Modellgenauigkeit: {accuracy:.2%}")  
    #col1, col2, col3 = st.columns(3)

    #with col1:
        #st.pyplot(cm_fig)

    #with col2:
        #st.pyplot(tsne_fig)

    #with col3:
        #st.pyplot(dist_fig)

    st.info("🔄 Starte Clustering der Gesichtsembeddings...")

    with st.spinner("🔍 Gesichter werden analysiert..."):
        labels, paths = model.cluster_faces(cnn, all_images)
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    st.success(f"✅ Anzahl erkannter Personen (Cluster): {num_clusters}")

    # Beispielbilder pro Cluster anzeigen
    TARGET_SIZE = (200, 200)  # Breite, Höhe

    st.subheader("📂 Bilder der Erkannten Person (max. 5)")

    cluster_map = {}
    for label, path in zip(labels, paths):
        if label not in cluster_map:
            cluster_map[label] = []
        if len(cluster_map[label]) < 5:
            cluster_map[label].append(path)

    for label, image_paths in cluster_map.items():
        label_name = f"🕵️‍♂️ Unbekannt" if label == -1 else f"👤 Person {label}"
        st.markdown(f"### {label_name}")
        cols = st.columns(len(image_paths))
        for i, img_path in enumerate(image_paths):
            try:
                img = Image.open(img_path).resize(TARGET_SIZE)
                cols[i].image(img)
            except Exception as e:
                st.warning(f"Fehler beim Laden von Bild: {img_path}")

    if not os.path.exists("face1.jpg"):
        st.warning("Bitte zuerst ein Bild unter 📷 Kamera aufnehmen und speichern.")
    else:
        st.info("🔄 Suche nach ähnlichen Gesichtern...")

        with st.expander("📄 Gesichtserkennung und Caching"):
            st.markdown("""
            ### Überblick über den Prozess der Gesichtserkennung

            **1. Gesicht vergleichen:**  
            Ein Eingabegesicht wird mit den Bildern im Datensatz verglichen, um die ähnlichsten Gesichter zu identifizieren.

            **2. Zwischenspeichern (Caching):**  
            Um die Effizienz zu steigern und wiederholte Berechnungen zu vermeiden, werden Ergebnisse im Cache gespeichert.

            **3. Vorverarbeitung und Embedding:**  
            Bilder werden in Tensors umgewandelt und durch das Modell geführt, um **Embeddings** (numerische Repräsentationen der Gesichter) zu berechnen.

            **4. Ähnlichkeitsberechnung:**  
            Die Ähnlichkeit zwischen dem Eingabegesicht und den Gesichtern im Datensatz wird durch die Berechnung der **Euklidischen Distanz** zwischen ihren Embeddings gemessen.

            **5. Ergebnisse anzeigen:**  
            Die ähnlichsten Gesichter werden angezeigt, und falls ein Fehler auftritt, wird eine entsprechende Fehlermeldung ausgegeben.
            """)
        with st.spinner("Vergleiche Gesicht mit Datensatz..."):
            similar_faces = model.find_similar_faces(cnn, "face1.jpg", all_images[:1000000], top_k=5)

        st.subheader("👥 Ähnliche Gesichter")
        if not similar_faces:
            st.warning("Keine ähnlichen Gesichter gefunden.")
        else:
            cols = st.columns(5)
            for i, (path, dist) in enumerate(similar_faces):
                try:
                    img = Image.open(path).resize(TARGET_SIZE)
                    cols[i % 5].image(img, caption=f"Distanz: {dist:.2f}")
                except Exception as e:
                    st.error(f"Fehler beim Laden von Bild: {path}")

        with st.expander("📄 Projekt-Dokumentation (Kurzfassung)"):
            st.markdown("""
            ### 🔧 Verwendete Technologien
            - **Streamlit**: Für das UI und die einfache Web-Darstellung
            - **PyTorch**: Training und Nutzung eines CNN zur Extraktion von Gesichtsembeddings
            - **OpenCV**: Bildverarbeitung (z. B. Face Detection)
            - **Scikit-Learn**: Clustering (DBSCAN), Ähnlichkeitssuche (Cosine/Euclidean), Visualisierung (t-SNE)
            - **PIL / Matplotlib / Seaborn**: Bildanzeige und Visualisierung

            ### 🚀 Performance-Optimierungen
            - `@st.cache_data`: Bilderpfade werden nur einmal eingelesen → spart Ladezeit
            - **Vorgefertigtes Modell**: Einmaliges Training + Speicherung → kein Training bei jedem Start
            - **Limitierung der Bildanzahl** (`max_images`) bei Analyse → Vermeidung von RAM-Überlastung
            - **Verkleinerung der Bilder** auf 64×64 oder 150×200 Pixel → schnellere Verarbeitung
            - **Ähnlichkeitssuche mit Caching** → wiederholte Abfragen schneller

            ### 🧠 Funktionsübersicht
            - **Bildpfade laden** (`load_all_image_paths`)
            - **Gesichtserkennung** mit OpenCV-Haarcascade
            - **Clustering** der Embeddings mit DBSCAN zur Gruppierung ähnlicher Personen
            - **Suche ähnlicher Gesichter** basierend auf CNN-Features
            - **Visualisierung**: Konfusionsmatrix, t-SNE, Histogramme
            - **Caching von Embeddings und Ergebnissen** zur Beschleunigung bei wiederholtem Zugriff
            """)


elif st.session_state.page == "Analyse Gökhan":
    st.write("Hier findet die Datenanalyse statt...")

    if st.button("Starte Gesichtserkennung"):
        with st.spinner("Verarbeite Bilder..."):
            originalbild, bild_grau, bild_rgb, gesicht_klein, zufallsbilder, datensatz, bilder, rate  = modelg.verarbeite_gesicht()

            st.write(originalbild)
            # Bilder vorher korrekt vorbereiten
            originalbild_rgb = cv2.cvtColor(originalbild, cv2.COLOR_BGR2RGB)
            bild_rgb = cv2.cvtColor(bild_rgb, cv2.COLOR_BGR2RGB)  # Falls nötig

            fig, axs = plt.subplots(1, 4, figsize=(20, 5))  # 1 Zeile, 4 Spalten

            # Originalbild
            axs[0].imshow(originalbild_rgb)
            axs[0].set_title("Originalbild")
            axs[0].axis("off")

            # Graubild
            axs[1].imshow(bild_grau, cmap="gray")
            axs[1].set_title("Bild in grauem Farbton")
            axs[1].axis("off")

            # RGB mit Gesichtserkennung
            axs[2].imshow(bild_rgb)
            axs[2].set_title("Farbe mit Gesichtserkennung")
            axs[2].axis("off")

            # Gesicht in 75x100
            axs[3].imshow(gesicht_klein, interpolation='nearest')
            axs[3].set_title("75x100 gespeichert")
            axs[3].axis("off")

            plt.tight_layout()
            st.pyplot(fig)

            # Beispiel: zufallsbilder
            fig = plt.figure(figsize=(15, 5))
            for i, bild in enumerate(zufallsbilder):
                plt.subplot(2, 5, i + 1)
                plt.imshow(bild)
                plt.axis("off")
                plt.title(f"Bild {i+1}")
            plt.tight_layout()

            st.pyplot(fig)

            st.subheader("📊 Zusammenfassende Auswertung")

            st.write(f"👥 **Anzahl der Gesichter insgesamt im Datensatz:** {datensatz}")
            st.write(f"🖼️ **Anzahl der erkannten Bilder:** {bilder}")
            st.write(f"✅ **Erkennungsrate:** {rate:.2f} %")

    if st.button("Starte Gesichtserkennung"):
        with st.spinner("Verarbeite Bilder..."):
            originalbild, bild_grau, bild_rgb, gesicht_klein = modelg.verarbeite_gesicht()

        if originalbild is not None:
            st.subheader("📸 Originalbild mit Gesichtserkennung")
            st.image(originalbild, channels="BGR", caption="Originalbild")

            st.subheader("🖤 Bild in Graustufen")
            st.image(bild_grau, clamp=True, caption="Graues Bild", channels="GRAY")

            st.subheader("🎯 Bild in Farbe mit Gesicht")
            st.image(bild_rgb, caption="RGB + Gesicht markiert")

            st.subheader("🔍 Gesicht in 75x100")
            st.image(gesicht_klein, caption="Ausgeschnittenes Gesicht (75x100)", width=150)
        else:
            st.warning("Kein passendes Bild mit genau einem Gesicht gefunden.")


elif st.session_state.page == "Face Recognition":
    face_recognition_page.app()


elif st.session_state.page == "Kamera":
    st.write("Hier könntest du ein Bild mit deiner Webcam aufnehmen...")
    camera.app()

    if st.session_state.get("captured_image") is not None:
        st.image(st.session_state.captured_image, caption="Bild aus Kamera", channels="BGR")

        if st.button("💾 Bild speichern"):
            import cv2
            cv2.imwrite("snapshot.jpg", st.session_state.captured_image)
            st.success("Bild unter snapshot.jpg gespeichert.")
