import streamlit as st
import base64

def image_to_base64(path):
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def app():
    st.markdown("""
    <div style="border-top: 4px solid #00bfff; padding-top: 10px; margin-bottom: 20px;">
        <h2 style="color: white;">Gesichtserkennung</h2>
    </div>
""", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2]) 

    with col1:
        img_base64 = image_to_base64("gesichtserkennung.jpg")
        st.markdown(f"""
            <div style="text-align: center;">
                <img src="data:image/jpeg;base64,{img_base64}" style="height:570px; border-radius:10px;"><br>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("<h2>Projekt von Sergej, Volker und Gökhan - Data Science</h2>", unsafe_allow_html=True)
        st.markdown("""
            <div style="background-color: #1e1e1e; padding: 30px; border-radius: 10px;">
                <h2 style="color: #00ffcc;">Einführung</h2>
                <p style="color: white; font-size: 16px;">
                    Gesichtserkennung ist eine der spannendsten Anwendungen von <b>Computer Vision</b>.
                    Dabei werden menschliche Gesichter in Bildern oder Videos erkannt und analysiert. 
                    Typische Einsatzbereiche sind z. B. <i>Authentifizierung, Sicherheitsüberwachung</i> 
                    oder <i>Emotionserkennung</i>.
                </p>
                <ul style="color: white;">
                    <li>Computer Vision</li>
                        <li style="list-style-type: none; padding-left: 20px;">
                            <span style="font-size: 15px; color: #ccc;">
                                Erklärung: Computer Vision ist ein Teilgebiet der künstlichen Intelligenz, das es Computern ermöglicht, Bilder und Videos zu "sehen", zu analysieren und zu verstehen.
                            </span>
                        </li>
                    <li>Datensatz/-sätze erklären</li>
                        <li style="list-style-type: none; padding-left: 20px;">
                            <span style="font-size: 15px; color: #ccc;">
                                Erklärung: Die verwendeten Datensätze bestehen aus Gesichtsbildern, die in Ordnern nach Personen sortiert sind. Sie werden mithilfe des FaceDataset-Objekts geladen, vorverarbeitet und in Tensoren umgewandelt, die vom Modell gelesen werden können.
                            </span>
                        </li>
                    <li>Data Augmentation</li>
                        <li style="list-style-type: none; padding-left: 20px;">
                            <span style="font-size: 15px; color: #ccc;">
                                Erklärung: Durch Techniken wie Bilddrehung, Spiegelung oder Helligkeitsanpassung kann künstlich die Vielfalt im Trainingsdatensatz erhöht werden. Dies verbessert die Generalisierungsfähigkeit des Modells.
                            </span>
                        </li>
                    <li>Aktueller Stand der Technik</li>
                        <li style="list-style-type: none; padding-left: 20px;">
                            <span style="font-size: 15px; color: #ccc;">
                                Erklärung: Moderne Gesichtserkennung basiert meist auf tiefen neuronalen Netzen wie CNNs (Convolutional Neural Networks), oft mit vortrainierten Backbones wie VGGFace, ResNet oder MobileNet. In diesem Projekt nutzen wir ein einfaches CNN-Modell.
                            </span>
                        </li>
                </ul>
            </div>
            <br>
        """, unsafe_allow_html=True)

    left, right = st.columns([1, 1])
    with left:
        st.markdown("""
            <div style="background-color: #1e1e1e; padding: 20px; border-radius: 10px;">
                <h3 style="color: #00bfff;">Arbeitsweise & Learnings</h3>
                <ul style="color: white; font-size: 15px;">
                    <li>Arbeitsweise</li>
                        <li style="list-style-type: none; padding-left: 20px;">
                            <span style="font-size: 15px; color: #ccc;">
                                Erklärung: Der Workflow bestand aus Datenvorbereitung (inkl. Preprocessing und Caching), Modelltraining, Gesichtserkennung und Visualisierung. Die Anwendung wurde mit Streamlit gebaut und modularisiert.
                            </span>
                        </li>
                    <li>Hindernisse</li>
                        <li style="list-style-type: none; padding-left: 20px;">
                            <span style="font-size: 15px; color: #ccc;">
                                Erklärung: Hauptprobleme traten bei defekten Bilddateien, unvollständigen Datenstrukturen und der Modellinitialisierung auf. Diese wurden durch Fehlerbehandlung und Datenvalidierung gelöst.
                            </span>
                        </li>
                    <li>Auffälligkeiten</li>
                        <li style="list-style-type: none; padding-left: 20px;">
                            <span style="font-size: 15px; color: #ccc;">
                                Erklärung: Einige Bilder waren extrem groß oder beschädigt. Das wurde bei der Preprocessing-Phase abgefangen und die Bilder wurden übersprungen.
                            </span>
                        </li>
                    <li>Learnings</li>
                        <li style="list-style-type: none; padding-left: 20px;">
                            <span style="font-size: 15px; color: #ccc;">
                                Erklärung: Effiziente Bildverarbeitung, Caching mit Pickle, sowie die Bedeutung robuster Vorverarbeitung und modularer Code-Strukturen wurden intensiv gelernt und umgesetzt.
                            </span>
                        </li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with right:
        img_base64 = image_to_base64("gesichtserkennung1.jpg")
        st.markdown(f"""
            <div style="text-align: center;">
                <img src="data:image/jpeg;base64,{img_base64}" style="height:440px; border-radius:10px;"><br>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([1, 1])
    with left:
        img_base64 = image_to_base64("gesichtserkennung2.jpg")
        st.markdown(f"""
            <div style="text-align: center;">
                <img src="data:image/jpeg;base64,{img_base64}" style="height:440px; border-radius:10px;"><br>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    with right:
        st.markdown("""
            <div style="background-color: #1e1e1e; padding: 20px; border-radius: 10px;">
                <h3 style="color: #00bfff;">Performance & Optimierung</h3>
                <ul style="color: white; font-size: 15px;">
                    <li>Warum funktioniert es gut?</li>
                        <li style="list-style-type: none; padding-left: 20px;">
                            <span style="font-size: 15px; color: #ccc;">
                                Erklärung: Das Modell erzielt gute Resultate bei klaren, gut belichteten Bildern. Die Kombination aus Caching, Vorverarbeitung und CNN sorgt für stabile Performance.
                            </span>
                        </li>
                    <li>Warum schlecht?</li>
                        <li style="list-style-type: none; padding-left: 20px;">
                            <span style="font-size: 15px; color: #ccc;">
                                Erklärung: Die Genauigkeit leidet bei schlechten Lichtverhältnissen, niedriger Auflösung oder ungewöhnlichen Gesichtswinkeln. Auch die geringe Tiefe des Netzwerks limitiert die Erkennungsleistung.
                            </span>
                        </li>
                    <li>Was kann man verbessern?</li>
                        <li style="list-style-type: none; padding-left: 20px;">
                            <span style="font-size: 15px; color: #ccc;">
                                Erklärung: Verwendung eines vortrainierten Modells (z. B. ResNet oder VGGFace), umfangreichere Datensätze und gezieltere Augmentation könnten die Performance deutlich steigern.
                            </span>
                        </li>
                    <li>Was nicht?</li>
                        <li style="list-style-type: none; padding-left: 20px;">
                            <span style="font-size: 15px; color: #ccc;">
                                Erklärung: Einfache lineare Skalierung bringt kaum Verbesserung. Ohne zusätzliche Daten oder tiefere Netzarchitekturen bleibt die Leistung begrenzt.
                            </span>
                        </li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([1, 1])
    with left:
        st.markdown("""
            <div style="background-color: #1e1e1e; padding: 20px; border-radius: 10px;">
                <h3 style="color: #00bfff;">Datenstrategie</h3>
                <ul style="color: white; font-size: 15px;">
                    <li>Umgang mit fehlenden Daten</li>
                        <li style="list-style-type: none; padding-left: 20px;">
                            <span style="font-size: 15px; color: #ccc;">
                                Erklärung: Hier gab es keine Probleme
                            </span>
                        </li>
                    <li>Data Augmentation</li>
                        <li style="list-style-type: none; padding-left: 20px;">
                            <span style="font-size: 15px; color: #ccc;">
                                Erklärung: (optional) Hätte eingesetzt werden können, um die Trainingsbasis künstlich zu vergrößern und das Modell robuster gegenüber neuen Bildern zu machen.
                            </span>
                        </li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    with right:
        img_base64 = image_to_base64("gesichtserkennung3.jpg")
        st.markdown(f"""
            <div style="text-align: center;">
                <img src="data:image/jpeg;base64,{img_base64}" style="height:440px; border-radius:10px;"><br>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

