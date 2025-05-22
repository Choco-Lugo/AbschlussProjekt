import os
import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE
import pickle
import hashlib
import seaborn as sns
import gdown
import streamlit as st
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

class SimpleFaceCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 14 * 14, 128)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class FaceDataset(Dataset):
    def __init__(self, root_dir):
        if root_dir == "__dummy__":
            return
        self.data = []
        self.labels = []
        self.label_map = {}
        label_counter = 0

        for person in os.listdir(root_dir):
            person_dir = os.path.join(root_dir, person)
            if not os.path.isdir(person_dir):
                continue
            print(f"📁 Person-Ordner: {person_dir}")
            if person not in self.label_map:
                self.label_map[person] = label_counter
                label_counter += 1
            for file in os.listdir(person_dir):
                path = os.path.join(person_dir, file)
                print(f"🔍 Datei gefunden: {path}")
                if not file.lower().endswith(('.jpg', '.jpeg', '.png')):  
                    print("   ⛔ Kein Bildformat")
                    continue
                if cv2.imread(path) is None:
                    print("   ⚠️ Ungültiges Bild, wird übersprungen.")
                    continue  
                self.data.append(path)
                self.labels.append(self.label_map[person])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        image = cv2.imread(img_path)
        image = cv2.resize(image, (64, 64))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        label = self.labels[idx]
        return image, label

def _build_drive_client() -> GoogleDrive:
    """Erzeuge einen autorisierten PyDrive-Client aus st.secrets."""
    cfg = {
        "client_config_backend": "settings",
        "client_config": {
            "client_id": st.secrets["google_drive"]["client_id"],
            "client_secret": st.secrets["google_drive"]["client_secret"],
            "redirect_uri": "urn:ietf:wg:oauth:2.0:oob",
        },
        "save_credentials": True,
        "save_credentials_backend": "file",
        "save_credentials_file": "drive_creds.json",
        "get_refresh_token": True,
    }

    gauth = GoogleAuth(settings=cfg)
    # Tokens aus secrets einsetzen
    gauth.credentials = GoogleAuth(settings=cfg).oauth2client.client.OAuth2Credentials(
        access_token=None,
        client_id     = st.secrets["google_drive"]["client_id"],
        client_secret = st.secrets["google_drive"]["client_secret"],
        refresh_token = st.secrets["google_drive"]["refresh_token"],
        token_expiry  = None,
        token_uri     = "https://oauth2.googleapis.com/token",
        user_agent    = "streamlit-cloud/1.0",
        scopes        = ["https://www.googleapis.com/auth/drive.readonly"],
    )
    return GoogleDrive(gauth)

def download_from_gdrive(file_id: str, output_path: str) -> None:
    """Lädt eine öffentliche oder für den Service-Account freigegebene Datei."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False, fuzzy=True)

@st.cache_data(show_spinner="📂 Lade Bilder aus Google Drive...")
def load_all_images_recursive(folder_id, image_size=(150, 150)):
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)

    def get_all_images_from_folder(fid):
        image_list = []
        file_list = drive.ListFile({'q': f"'{fid}' in parents and trashed=false"}).GetList()

        for file in file_list:
            if file['mimeType'] == 'application/vnd.google-apps.folder':
                # Ordner → rekursiv abrufen
                image_list.extend(get_all_images_from_folder(file['id']))
            elif file['title'].lower().endswith(('.jpg', '.jpeg', '.png')):
                downloaded = file.GetContentIOBuffer()
                img = Image.open(downloaded).convert("RGB")
                img = img.resize(image_size)
                image_list.append(np.array(img))
        return image_list

    return get_all_images_from_folder(folder_id)

def train_model(root_dir="modelV/VGG-Face2/data/vggface2_train/train", epochs=2):
    model = SimpleFaceCNN()
    model.classifier.add_module("output", nn.Linear(128, len(os.listdir(root_dir))))  
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    dataset = FaceDataset(root_dir)

    if len(dataset) == 0:
        raise ValueError("❌ Das Dataset enthält keine gültigen Bilder. Bitte überprüfe den Pfad und die Dateien.")
    
    dataloader = DataLoader(dataset, batch_size=16, num_workers=0, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for images, labels in dataloader:
            optimizer.zero_grad()
            features = model.features(images)
            features = model.classifier[:-1](features.view(images.size(0), -1))  
            outputs = model.classifier[-1](features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    return model

def evaluate_model(model, dataset, embedding_cache_path=None):
    if embedding_cache_path and os.path.exists(embedding_cache_path):
        embeddings, y_true, y_pred = load_embeddings(embedding_cache_path)
    else:
        y_true, y_pred, features = [], [], []

        model.eval()
        with torch.no_grad():
            for imgs, labels in DataLoader(dataset, batch_size=16, num_workers=0):
                feats = model.features(imgs).view(imgs.size(0), -1)
                outputs = model.classifier(feats)
                preds = outputs.argmax(dim=1)

                y_true.extend(labels.numpy())
                y_pred.extend(preds.numpy())
                features.append(feats)

        embeddings = torch.cat(features).numpy()

        if embedding_cache_path:
            save_embeddings(embeddings, y_true, y_pred, embedding_cache_path)

    # Genauigkeit
    accuracy = accuracy_score(y_true, y_pred)

    # Konfusionsmatrix
    cm_fig, cm_ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred))
    disp.plot(ax=cm_ax)
    cm_ax.set_title("Konfusionsmatrix")

    # t-SNE
    tsne_fig, tsne_ax = plt.subplots(figsize=(8, 6))
    reduced = TSNE(n_components=2, random_state=42).fit_transform(embeddings)
    sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=y_true, ax=tsne_ax, palette="tab10", legend=False)
    tsne_ax.set_title("t-SNE der Gesichtsembeddings")

    # Klassenverteilung
    dist_fig, dist_ax = plt.subplots(figsize=(8, 4))
    sns.histplot(y_true, bins=len(set(y_true)), ax=dist_ax)
    dist_ax.set_title("Klassenverteilung im Training")

    return accuracy, cm_fig, tsne_fig, dist_fig


def count_faces_in_image(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    img = cv2.imread(image_path)
    if img is None:
        return 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return len(faces)

def find_similar_faces(model, input_img_path, dataset_paths, top_k=5):
    cache_file = _get_cache_filename(input_img_path, dataset_paths)
    cached_result = _load_from_cache(cache_file)
    if cached_result:
        return cached_result

    def preprocess(path):
        try:
            with Image.open(path) as img:
                if img.size[0] > 2000 or img.size[1] > 2000:  # Beispiel: Max 2000x2000
                    print(f"Bild zu groß: {path}, übersprungen.")
                    return None
        except Exception as e:
            print(f"Fehler beim Öffnen von {path}: {e}")
            return None

        img = cv2.imread(path)
        if img is None:
            return None
        img = cv2.resize(img, (64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        return tensor

    input_tensor = preprocess(input_img_path)
    if input_tensor is None:
        return []

    model.eval()
    with torch.no_grad():
        input_embedding = model(input_tensor).squeeze().numpy()

    embeddings = []
    valid_paths = []

    for path in dataset_paths:
        tensor = preprocess(path)
        if tensor is None:
            continue
        with torch.no_grad():
            emb = model(tensor).squeeze().numpy()
        embeddings.append(emb)
        valid_paths.append(path)

    distances = [np.linalg.norm(input_embedding - emb) for emb in embeddings]
    sorted_indices = np.argsort(distances)[:top_k]
    result = [(valid_paths[i], distances[i]) for i in sorted_indices]

    _save_to_cache(cache_file, result)
    return result

def extract_embeddings(model, image_paths):
    embeddings = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0

        with torch.no_grad():
            embedding = model(tensor).squeeze().numpy()
            embeddings.append(embedding)

    return np.array(embeddings)

def cluster_faces(model, image_paths, max_images=500):
    embeddings = []
    paths = []
    count = 0

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0

        with torch.no_grad():
            embedding = model(tensor).squeeze().numpy()
            embeddings.append(embedding)
            paths.append(img_path)

        count += 1
        if count >= max_images:
            break

    clusterer = DBSCAN(eps=5.0, min_samples=2, metric='euclidean')
    labels = clusterer.fit_predict(embeddings)

    return labels, paths

def cluster_farben(labels, paths):
    cluster_colors = {}
    for label, path in zip(labels, paths):
        if label == -1:
            continue
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mean_color = np.mean(img.reshape(-1, 3), axis=0)
        if label not in cluster_colors:
            cluster_colors[label] = []
        cluster_colors[label].append(mean_color)

    fig, ax = plt.subplots(figsize=(10, 2))
    return fig, ax, cluster_colors

def cluster_dimen(cnn, paths):
    embeddings = []
    for path in paths:
        img = cv2.imread(path)
        img = cv2.resize(img, (64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        with torch.no_grad():
            emb = cnn(tensor).squeeze().numpy()
            embeddings.append(emb)
    embeddings = np.array(embeddings)

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)

    fig, ax = plt.subplots()
    return fig, ax, reduced

def save_model(model, path="face_model.pth"):
    torch.save(model.state_dict(), path)

def load_model(path="face_model.pth"):
    model = SimpleFaceCNN()
    model.classifier.add_module("output", nn.Linear(128, 8631))  
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

def _get_cache_filename(input_img_path, dataset_paths, cache_dir="cache"):
    os.makedirs(cache_dir, exist_ok=True)
    key = input_img_path + "|" + "|".join(dataset_paths)
    hashed = hashlib.md5(key.encode()).hexdigest()
    return os.path.join(cache_dir, f"similar_faces_{hashed}.pkl")

def _load_from_cache(cache_path):
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    return None

def _save_to_cache(cache_path, data):
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)

def save_dataset(dataset, path="cached_dataset.pkl"):
    with open(path, 'wb') as f:
        pickle.dump((dataset.data, dataset.labels, dataset.label_map), f)

def load_dataset(path="cached_dataset.pkl"):
    with open(path, 'rb') as f:
        data, labels, label_map = pickle.load(f)

    dataset = FaceDataset("__dummy__")  
    dataset.data = data
    dataset.labels = labels
    dataset.label_map = label_map
    return dataset

def save_embeddings(embeddings, y_true, y_pred, path="cached_embeddings.pkl"):
    with open(path, 'wb') as f:
        pickle.dump((embeddings, y_true, y_pred), f)

def load_embeddings(path="cached_embeddings.pkl"):
    with open(path, 'rb') as f:
        embeddings, y_true, y_pred = pickle.load(f)
    return embeddings, y_true, y_pred