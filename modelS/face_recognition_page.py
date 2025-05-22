import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA # type: ignore

import modelS.face_detection as face_detection # type: ignore
import modelS.face_recognition as face_recognition # type: ignore

def app():

    st.write('Face Recognition ist eine Technik der Computer Vision, mit der ein Algorithmus an Hand von Ähnlichkeitsmerkmalen ein noch ungesehenes Bild mit einem Gesicht einer Person zuzuordnen.')

    st.write('')
    st.write('')
    st.write('')
    st.title('Modellübersicht')
    
    st.write('Zwei verschiedene Modelle wurden entwickelt:')
    
    col_S1, col_S2, col_S3, col_S4 = st.columns([1.3, 2, 2, 3]) 
    
    with col_S1:
        st.info('Komponente')
        st.write('Dimemsionsreduktion:')
        st.write('Estimator:')
        st.divider() 

        st.write('3 Personen:')
        st.write('5 Personen:')
        st.write('8 Personen:')
        st.write('19 Personen:')
    
    with col_S2:
        st.info('Modell A - Supervised Learning')
        st.write('Principal Component Analysis (PCA)')
        st.write('Support-Vector-Machine (SVM)')
        st.divider() 
        st.write('A1: 96,16 % Accuracy')
        st.write('A2: 90,79 % Accuracy')
        st.write('A3: 85,56 % Accuracy')
        st.write('A4: 74,87 & Accuracy')
        
    with col_S3:
        st.info('Modell B - No Learning')
        st.write('Principal Component Analysis (PCA)')
        st.write('Cosine Similarity (CS)')
        st.divider()
        st.write('B1: 96,15 % Accuracy')
        st.write('B2: 90,79 % Accuracy')
        st.write('B3: 87,78 % Accuracy')
        st.write('B4: 77,54 % Accuracy')

    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    
    st.image(image='modelS/plots/numer_of_pictures.png',
            caption='Verteilung der Bilder auf Personen im LFW-Datensatz',
            width=900)
    
    st.write(f'Da Personen mit vielen Bildern im Datensatz (rechter Rand der $x$-Achse) die Minderheit im Datensatz bilden, habe ich mittels Random Oversampling (ROS) die Anzahl an Samples von unterrepräsentiertern Klassen im Trainingsdatensatz vervielfältigen lassen.')    

        
    st.write('')
    st.write('')
    st.divider()
    st.title('Principal Component Analysis')
    st.write(f'PCA reduziert einen Datensatz auf $n$ Hauptkomponenten, bei gleichzeitiger Beibehaltung der maximalen Varianz der Daten.')
    
    st.image(image='https://upload.wikimedia.org/wikipedia/commons/f/f5/GaussianScatterPCA.svg',
            caption='Beispielbild einer PCA [Von Nicoguaro - Eigenes Werk, CC BY 4.0, https://commons.wikimedia.org/w/index.php?curid=46871195]',
            width=500)
    
    st.subheader('Beispielrechnung:')
    st.write(f'Ein farbiges Bild in der Größe $100x75$ Pixeln besitzt folgende Datenpunkte $x$:')
    st.latex(r'x = h \cdot w \cdot 3')
    st.write(f'Mit $h$ für die Höhe, $w$ für die Breite und $3$ für die Anzahl an Farben des Farbraums, in dem Bild abgespeichert ist (z.B. RGB = Rot, Gelb, Blau) ')
    st.write(f'Danach ergeben sich für $x$')
    st.latex(r'x = 100 \cdot 75 \cdot 3 = 7500 \cdot 3 = 22500')
    st.write('Komponenten pro Bild')
    st.write('')
    st.write('')
    st.write('')

    
    
    
    st.subheader('Eigen-Bilder')
    
    st.write(f'Reduktion der Bilder auf $n$-Komponenten. Die Inverse der PCA kann als Eigenbild (Darstellung der Eigenvektoren) dargestellt werden, um die Dimensionsreduktion zu veranschaulichen:')
    st.write(f"Die hier dargestellten Eigenbilder basieren auf einer PCA, die auf die 530 Bildern von George W. Bush aus dem LFW-Datensatz angewendet wird. Wird der Wert des Sliders für die $n$ Dimensionen auf 1 gesetzt, zeigt sich ein Gesicht von George W. Bush, das als 'Durchschnittsgesicht' dessen angesehen werden kann, das auf Grundlage der 530 Gesichtsbilder von George W. Bush entsteht.")
    X_Bush_uint8 = np.load('modelS/eigenfaces/Bush.npz')['Bush']
    X_Bush_float64 = X_Bush_uint8.reshape(X_Bush_uint8.shape[0], -1).astype(np.float64) / 255.0
    
    pic = st.slider('Wähle ein Bild aus:', 0, 500, 198, key=1)
    n_components = st.slider(f'Reduzierung auf $n$ Dimensionen:', 1, 500, 90, key=2)
    
    col_S5, col_S6, col_S7 = st.columns([1.5,1.5,2])
    with col_S5:
        image_4Darray = X_Bush_uint8[pic]
        fig = plt.figure()
        plt.title('Original-Bild')
        plt.xlabel(f'$w$')
        plt.ylabel(f'$h$')
        plt.imshow(image_4Darray)
        st.pyplot(fig)

        
    with col_S6:
        pca_show = PCA(n_components=n_components, whiten=True , random_state=42)
        X_2D_pca = pca_show.fit_transform(X_Bush_float64)
        X_2D_pca_inv = pca_show.inverse_transform(X_2D_pca)
        X_4D_pca_inv = X_2D_pca_inv[pic].reshape(100, 75, 3)
        print(X_4D_pca_inv)
        fig = plt.figure()
        plt.title('Eigen-Bild')
        plt.xlabel(f'$w$')
        plt.ylabel(f'$h$')
        plt.imshow(X_4D_pca_inv)
        st.pyplot(fig)
    
    st.divider()
    st.title('Modell A - Supervised Learning mit Support-Vector-Machine')
    st.subheader('PCA Explained Variance für LFW Datensatz')
    st.image(image='modelS/plots/pca.png',
            caption='PCA auf LFW Datensatz',
            width=600)
    st.write('')
    st.write('')
    st.write('')
        
    st.subheader('PCA Explained Variance für LFW Datensatz')
    
    st.write('Gridsearch-CV zur Auswahl der geeigneten Anzahl an Komponenten $n$ auf die mit PCA reduziert werden soll, sowie weitere Hyperparameter des SVM-Classifiers:')

    col_Sa1, col_Sa2 = st.columns([1,1])
    with col_Sa1:
        st.image(image='modelS/plots/FIG_bw_100x75_80_sca_pca_svm_2025-05-01_07-06-14.png',
                caption=f'Modell A - Schwarz-Weiß: Gridsearch-CV für PCA sowie SVM',
                width=600)
    with col_Sa2:
        st.image(image='modelS/plots/FIG_clr_100x75_80_sca_pca_svm_2025-05-02_22-21-35.png',
                caption=f'Modell A - RGB-Spektrum: Gridsearch-CV für PCA sowie SVM',
                width=600)
    
    st.write('Das Modell perfomt mit RGB-Bildern besser, als mit Schwarz-Weiß-Bildern. Daher habe ich die weiteren Untersuchungen lediglich mit RGB-Bildern vorgenommen.')
    st.write('')
    st.write('')
    st.write('')
    
    st.divider()
    st.title('Modell B - No Learning mit Cosine Similarity')
    st.write('(dt.: Kosinus Ähnlichkeit)')
    
    st.subheader('Mathematischer Hintergrund')
    st.write('Kosinus-Ähnlichkeit ist ein Maß für die Ähnlichkeit zweier Vektoren.\nDabei wird der Kosinus des Winkels zwischen beiden Vektoren bestimmt.')
    st.write('https://de.wikipedia.org/wiki/Kosinus-%C3%84hnlichkeit')
    st.write('')
    st.latex(r'Cosine \ Similarity = CS = cos(\theta) = \dfrac{A \cdot B}{||A|| \ ||B||}  \quad , \quad CS \in [-1, \ 1]')
    st.write('')
    st.write(f'$CS = -1$  bedeutet maximal großer Winkel zwischen den Datenpunkten, also sehr große Unähnlichkeit.')
    st.write(f'$CS = 0$  bedeutet quasi orthogonale Anordnung zwischen den Datenpunkten, deutet auf Unabhängigkeit hin.')
    st.write(f'$CS = 1$  bedeutet Winkel von quasi 0° zwischen den Datenpunkten, also sehr große Ähnlichkeit')
    st.write('')
    st.write('')

    st.subheader('Implementierung von Cosine Similarity')
    st.write('')
    st.info(f'Schritt 1:  CS Bild pro Bild berechnen')
    st.write('Für jedes Bild aus dem Testdatensatz X_test wird die CS zu jedem einzelnen Bild aus dem Trainingsdatensatz X_train berechnet:')
    st.write('(Zur besseren Veranschaulichung wurde für die folgenden Darstellungen ROS nicht angewandt)')
    st.write('')
    st.write('')
    
    
    col_Sb1, col_Sb2 = st.columns([1,1])
    with col_Sb1:
        st.image(image='modelS/plots/CS_perperson_Schroeder_everyone.png',
                caption=f'CS X_test von Gerhard Schröder mit X_train von allen',
                width=500)
    with col_Sb2:
        st.image(image='modelS/plots/CS_perperson_Schroeder_Bush.png',
                caption=f'CS X_test von Gerhard Schröder mit X_train von Geroge W. Bush',
                width=500)
    st.write('')
    st.write('')
    st.write('')
    st.write('')

    col_Sc1, col_Sc2, col_Sc3 = st.columns([1,3,1])
    with col_Sc2:
        st.image(image='modelS/plots/CS_perperson_Schroeder_Schroeder.png',
                caption=f'CS X_test von Gerhard Schröder mit X_train von Gerhard Schröder',
                width=600)
    st.write('')
    st.write('')

    st.divider()
    
    st.write('')

    st.info(f'Schritt 2:  CS pro Bild pro Person berechnen')
    st.write('Für jedes Bild aus dem Testdatensatz X_test wird die CS zu jeweils jeder Person aus dem Trainingsdatensatz X_train berechnet:')
    st.write('')
    st.write('')
    col_Sd1, col_Sd3 = st.columns([2,1])
    
    with col_Sd1:
        st.image(image='modelS/plots/CS_pic76.png',
                caption=f'',
                width=1100)
    with col_Sd3:
        st.image(image='modelS/plots/CS_dist_76.png',
                caption=f'Prediction for Test-picture 76',
                width=500)
    st.write('')
    
    st.divider()
    
    st.write('')
    
    col_Se1, col_Se4 = st.columns([2,1])
    with col_Se1:
        st.image(image='modelS/plots/CS_pic99.png',
                caption=f'',
                width=1100)
    with col_Se4:
        st.image(image='modelS/plots/CS_dist_99.png',
                caption=f'Prediction for Test-picture 99',
                width=500)
    
    st.write('')

    st.divider()

    st.write('')
    
    st.title('Optimierte Bedingungen und Hyperparameter der Modelle')
    st.write('')
    st.write('Unter den folgenden grundlegenden Bedingungen bei der Anwendung auf den Trainingsdatensatz und das Modells haben diese eine verbesserte Performance aufgewiesen:')
    
    col_1, col_2, col_3 = st.columns([1,1,1])
    with col_1:
        st.info('Grundlegend')
        st.write('LFW Dataset: color = True')
        st.write('ROS: True')
        st.write('PCA: whiten = True')
    with col_2:
        st.info('Modell A')
        st.write("SVM: class_weight = 'balanced', kernel = 'rbf'")
    with col_3:
        st.info('Modell B')
        st.write(r"CS: $CS_{person} = \sum_{i=1}^n CS_{person, i}^3$ $\quad$ , mit $i$ als Bild in $X_{train}$")
    
    st.write('')
    
    st.divider()
    st.title('Zusammenführen von Face Detection & Face Recognition')
    st.write('')
    st.write('')
    
    col_SL11, col_SR11 = st.columns([1,1])
    
    with col_SL11:
        selected_model = st.radio("Modell für Prediction",
                                [':rainbow[A1: PCA + SVM [96,15 % Accuracy]]', ':rainbow[A2: PCA + SVM [90,79 % Accuracy]]', ':rainbow[A3: PCA + SVM [85,56 & Accuracy]]', ':rainbow[A4: PCA + SVM [74,87 & Accuracy]]', ':rainbow[B1: PCA + CS [96,15 % Accuracy]]', ':rainbow[B2: PCA + CS [90,79 % Accuracy]]', ':rainbow[B3: PCA + CS [87,78 % Accuracy]]', ':rainbow[B4: PCA + CS [77,54 % Accuracy]]'],
                                captions=["3 Personen  [Colin Powell, George W Bush, Tony Blair]",
                                            "5 Personen  [Colin Powell, Donald Rumsfeld, George W Bush, Gerhard Schroeder, Tony Blair]",
                                            "8 Personen  [Ariel Sharon, Colin Powell, Donald Rumsfeld, George W Bush, Gerhard Schroeder, Hugo Chavez, Junichiro Koizumi, Tony Blair]",
                                            "19 Personen [Ariel Sharon, Arnold Schwarzenegger, Colin Powell, Donald Rumsfeld, George W Bush, Gerhard Schroeder, Gloria Macapagal Arroyo, Hugo Chavez, Jacques Chirac, Jean Chretien, Jennifer Capriati, John Ashcroft, Junichiro Koizumi, Laura Bush, Lleyton Hewitt, Luiz Inacio Lula da Silva, Serena Williams, Tony Blair, Vladimir Putin]",
                                            "3 Personen  [Colin Powell, George W Bush, Tony Blair]",
                                            "5 Personen  [Colin Powell, Donald Rumsfeld, George W Bush, Gerhard Schroeder, Tony Blair]",
                                            "8 Personen  [Ariel Sharon, Colin Powell, Donald Rumsfeld, George W Bush, Gerhard Schroeder, Hugo Chavez, Junichiro Koizumi, Tony Blair]",
                                            "19 Personen [Ariel Sharon, Arnold Schwarzenegger, Colin Powell, Donald Rumsfeld, George W Bush, Gerhard Schroeder, Gloria Macapagal Arroyo, Hugo Chavez, Jacques Chirac, Jean Chretien, Jennifer Capriati, John Ashcroft, Junichiro Koizumi, Laura Bush, Lleyton Hewitt, Luiz Inacio Lula da Silva, Serena Williams, Tony Blair, Vladimir Putin]"
                                            ]
                                ) 
    with col_SR11:
        if 'A1' in selected_model:
            filename_CM = "modelS/plots/CM_A_140.png"
        elif 'A2' in selected_model:
            filename_CM = "modelS/plots/CM_A_100.png"
        elif 'A3' in selected_model:
            filename_CM = "modelS/plots/CM_A_60.png"
        elif 'A4' in selected_model:
            filename_CM = "modelS/plots/CM_A_40.png"
        elif 'B1' in selected_model:
            filename_CM = "modelS/plots/CM_B_140.png"
        elif 'B2' in selected_model:
            filename_CM = "modelS/plots/CM_B_100.png"
        elif 'B3' in selected_model:
            filename_CM = "modelS/plots/CM_B_60.png"
        elif 'B4' in selected_model:
            filename_CM = "modelS/plots/CM_B_40.png"
        
        st.image(image=filename_CM,
                caption=f'Confusion Matrix des ausgewählten Modells',
                width=700)

        
    image_url = st.text_input('Link zum Bild', 'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fmosaicmagazine.com%2Fwp-content%2Fuploads%2F2021%2F11%2FColin-Powell-Main.jpg&f=1&nofb=1&ipt=169e05121ad6f6b016fe027bb3d348b58dc5e42293212039a8fdec666309a785')
    try: 
        og, og_marked, crop = face_detection.get_face(image_url)
        
        if 'PCA + SVM' in selected_model:

            col_SL, col_SR = st.columns([1,1])
            col_SL1, col_SL2, col_SR1, col_SR2 = st.columns([1,1,1,1])

            with col_SL:
                st.info('Face Detection')
                with col_SL1:
                    try:
                        fig = plt.figure()
                        plt.title('Original mit erkanntem Gesicht')
                        plt.xlabel(f'$w$')
                        plt.ylabel(f'$h$')
                        plt.imshow(og_marked)
                        st.pyplot(fig)
                    except:
                        pass
                
                with col_SL2:
                    try:
                        fig = plt.figure()
                        plt.title('Crop des erkanntes Gesichts')
                        plt.xlabel(f'$w$')
                        plt.ylabel(f'$h$')
                        plt.imshow(crop)
                        st.pyplot(fig)
                    except:
                        pass
                
            with col_SR:
                st.info('Face Recognition')
                with col_SR1:
                    with st.spinner('Berechne Ergebnisse...', show_time=True):
                        try:
                            pca, svm, X_2D = face_recognition.apply_model_A(selected_model=selected_model,
                                                                    crop=crop,
                                                                    )

                            X_2D_pca_inv = pca.inverse_transform(X_2D)
                            X_4D_pca_inv = X_2D_pca_inv.reshape(100, 75, 3)
                            fig = plt.figure()
                            plt.title('Eigen-Bild')
                            plt.xlabel(f'$w$')
                            plt.ylabel(f'$h$')
                            plt.imshow(X_4D_pca_inv)
                            st.pyplot(fig)
                        except:
                            pass
                        
                    with col_SR2:
                        try:
                            y_id = np.load('modelS/dataset/Target_ID.npy')
                            y_names = np.load('modelS/dataset/Target_Names.npy')
                            
                            y_pred = svm.predict(X_2D)
                            
                            mask = np.isin(np.unique(y_id), y_pred)
                            
                            st.write(f'Prediction:')       
                            st.write(y_names[mask])
                        except:
                            pass
        
            
        
        elif 'PCA + CS' in selected_model:
                        
            col_SL, col_SR = st.columns([1,1])
            col_SL1, col_SL2, col_SR1, col_SR2 = st.columns([1,1,1,1])

            with col_SL:
                st.info('Face Detection')
                with col_SL1:
                    try:
                        fig = plt.figure()
                        plt.title('Original mit erkanntem Gesicht')
                        plt.xlabel(f'$w$')
                        plt.ylabel(f'$h$')
                        plt.imshow(og_marked)
                        st.pyplot(fig)
                    except:
                        pass
                
                with col_SL2:
                    try:
                        fig = plt.figure()
                        plt.title('Crop des erkanntes Gesichts')
                        plt.xlabel(f'$w$')
                        plt.ylabel(f'$h$')
                        plt.imshow(crop)
                        st.pyplot(fig)
                    except:
                        pass
                
            with col_SR:
                st.info('Face Recognition')
                with col_SR1:
                    with st.spinner('Berechne Ergebnisse...', show_time=True):
                        try:
                            
                            pca, X_2D, X_train_pca, y_train, y_all, relevant_labels, target_names_all, n_targets = face_recognition.preprocess_cs(selected_model=selected_model,
                                                                                                                crop=crop
                                                                                                                )

                            X_2D_pca_inv = pca.inverse_transform(X_2D)
                            X_4D_pca_inv = X_2D_pca_inv.reshape(100, 75, 3)
                            fig = plt.figure()
                            plt.title('Eigen-Bild')
                            plt.xlabel(f'$w$')
                            plt.ylabel(f'$h$')
                            plt.imshow(X_4D_pca_inv)
                            st.pyplot(fig)
                            
                        except:
                            pass
                    
                with col_SR2:
                    try:

                        mask = face_recognition.calculate_cs(X_2D=X_2D,
                                                    X_train_pca=X_train_pca,
                                                    y_train=y_train,
                                                    y_all=y_all,
                                                    relevant_labels=relevant_labels,
                                                    n_targets=n_targets)

                        st.write(f'Prediction:')       
                        st.write(target_names_all[mask])
                    except:
                        pass
            
        else:
            st.write("Kein Modell ausgewählt.") 
            
    except:
        st.info('Kein Gesicht erkannt.')