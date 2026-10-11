import json
import os
import torch
import open_clip
import numpy as np
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "xlm-roberta-base-ViT-B-32"
pretrained = "laion5b_s13b_b90k"

model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
model = model.to(device)
model.eval()
tokenizer = open_clip.get_tokenizer(model_name)

client = QdrantClient(path="./qdrant_db")
VECTOR_SIZE = 512

COLLECTIONS = {
    "visual": "variant_1_visual",
    "text_desc": "variant_2_text_desc",
    "hybrid": "variant_3_hybrid"
}

for coll_name in COLLECTIONS.values():
    client.recreate_collection(
        collection_name=coll_name,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
    )

def encode_image(img_path: str) -> np.ndarray:
    """Generuje znormalizowany wektor cech wizualnych z obrazu."""
    image = Image.open(img_path).convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = model.encode_image(tensor)
        feats /= feats.norm(dim=-1, keepdim=True)
    return feats.squeeze(0).cpu().numpy()

def encode_text(text: str) -> np.ndarray:
    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats /= feats.norm(dim=-1, keepdim=True)
    return feats.squeeze(0).cpu().numpy()

ALLOWED_SUBFOLDERS = {"charts", "photos", "app_screenshots"}

def is_in_allowed_subfolder(path_str: str) -> bool:
    normalized_path = path_str.replace("\\", "/").lower()
    path_parts = set(normalized_path.split("/"))
    return not ALLOWED_SUBFOLDERS.isdisjoint(path_parts)

json_path = r"C:\Users\mateu\Desktop\Praktyki_altar\21_Test_mistralAI\descriptions.json"

with open(json_path, "r", encoding="utf-8") as f:
    dataset = json.load(f)

points_vis = []
points_txt = []
points_hyb = []
point_id = 0

for item in dataset:
    full_path = item.get("full_path", "")
    
    if not is_in_allowed_subfolder(full_path):
        continue

    file_name = item.get("file_name", "")
    analysis = item.get("analysis", {})

    img_type = analysis.get("type", "")
    summary = analysis.get("summary", "")
    visible_text = ", ".join(analysis.get("visibleText", []))
    keywords = ", ".join(analysis.get("keywords", []))

    text_to_embed = f"Typ: {img_type}. Opis: {summary}. Tekst na obrazie: {visible_text}. Słowa kluczowe: {keywords}."

    v_vis = encode_image(full_path)
    
    v_txt = encode_text(text_to_embed)
    
    v_hyb_raw = v_vis + v_txt
    v_hyb = v_hyb_raw / np.linalg.norm(v_hyb_raw)

    payload = {
        "file_name": file_name,
        "full_path": full_path,
        "type": img_type,
        "summary": summary
    }
    points_vis.append(PointStruct(id=point_id, vector=v_vis.tolist(), payload=payload))
    points_txt.append(PointStruct(id=point_id, vector=v_txt.tolist(), payload=payload))
    points_hyb.append(PointStruct(id=point_id, vector=v_hyb.tolist(), payload=payload))
    point_id += 1

client.upsert(collection_name=COLLECTIONS["visual"], points=points_vis)
client.upsert(collection_name=COLLECTIONS["text_desc"], points=points_txt)
client.upsert(collection_name=COLLECTIONS["hybrid"], points=points_hyb)

print(f"Zakończono tworzenie baz danych Qdrant dla {point_id} punktów.")