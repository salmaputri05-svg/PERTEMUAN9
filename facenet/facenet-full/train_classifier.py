import numpy as np, joblib
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

X = np.load("X_train.npy")
y = np.load("y_train.npy", allow_pickle=True)

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=10, gamma="scale", probability=True, class_weight="balanced"))
])

cv = min(2, len(X))   # 2-fold cross validation
scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")

print("Akurasi CV:", scores.mean(), "±", scores.std())

clf.fit(X, y)
joblib.dump(clf, "facenet_svm.joblib")
print("Model tersimpan.")
