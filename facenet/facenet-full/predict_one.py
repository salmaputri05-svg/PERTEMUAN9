import joblib, numpy as np
from utils_facenet import embed_from_path

clf = joblib.load("facenet_svm.joblib")

def predict_image(path, threshold=0.55):
    emb = embed_from_path(path)
    if emb is None:
        return "NO_FACE", 0.0
    proba = clf.predict_proba([emb])[0]
    idx = np.argmax(proba)
    label = clf.classes_[idx]
    conf = float(proba[idx])
    if conf < threshold:
        return "UNKNOWN", conf
    return label, conf

if __name__ == "__main__":
    test = "data/train/Saci/b1.jpg"
    print(predict_image(test))
