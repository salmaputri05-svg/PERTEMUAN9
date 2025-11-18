import os, glob, numpy as np, joblib
from collections import defaultdict
from utils_facenet import embed_from_path

clf = joblib.load("facenet_svm.joblib")

def predict_emb(emb):
    proba = clf.predict_proba([emb])[0]
    idx = np.argmax(proba)
    return clf.classes_[idx], float(proba[idx])

root = "data/val"
y_true, y_pred = [], []
stats = defaultdict(lambda: {"ok":0, "total":0})

for cls in sorted(os.listdir(root)):
    d = os.path.join(root, cls)
    if not os.path.isdir(d): 
        continue

    for img in glob.glob(os.path.join(d, "*")):
        emb = embed_from_path(img)
        if emb is None:
            continue

        pred, conf = predict_emb(emb)

        y_true.append(cls)
        y_pred.append(pred)

        stats[cls]["total"] += 1
        stats[cls]["ok"] += int(pred == cls)

# Akurasi
acc = np.mean([t == p for t, p in zip(y_true, y_pred)])
print("Akurasi total:", acc)

for c, st in stats.items():
    if st["total"] > 0:
        print(f"{c}: {st['ok']} / {st['total']} = {st['ok']/st['total']:.3f}")
