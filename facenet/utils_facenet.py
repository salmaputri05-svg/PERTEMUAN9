import torch, numpy as np, cv2
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Detector (alignment)
mtcnn = MTCNN(image_size=160, margin=20, post_process=True, device=device)

# Embedder (512 dim)
embedder = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def read_img_bgr(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Gagal membaca gambar: {path}")
    return img

def bgr_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

@torch.no_grad()
def face_align(img):
    pil = bgr_to_pil(img)
    return mtcnn(pil)

@torch.no_grad()
def embed_face_tensor(face):
    if face is None:
        return None
    face = face.unsqueeze(0).to(device)
    emb = embedder(face)
    return emb.squeeze(0).cpu().numpy()

@torch.no_grad()
def embed_from_path(path):
    img = read_img_bgr(path)
    aligned = face_align(img)
    if aligned is None:
        return None
    return embed_face_tensor(aligned)

def cosine_similarity(a, b):
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))
