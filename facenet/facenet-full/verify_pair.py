from utils_facenet import embed_from_path, cosine_similarity

img1 = "data/val/Pipi/a1.jpg"
img2 = "data/val/Saci/b1.jpg"

e1 = embed_from_path(img1)
e2 = embed_from_path(img2)

if e1 is None or e2 is None:
    print("Wajah tidak terdeteksi.")
else:
    sim = cosine_similarity(e1, e2)
    print("Similarity:", sim)
    print("MATCH" if sim >= 0.85 else "NO MATCH")
