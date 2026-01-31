import os
import sys
import json
import gc
import numpy as np
import torch
from dotenv import load_dotenv

from VideoHandler import VideoHandler
from MTCNNFaceExtractor import MTCNNFaceExtractor
from FacenetFeatureExtractor import FacenetFeatureExtractor
from HDBScanClusterer import HDBScanClusterer

load_dotenv()

GROUND_TRUTH_BASE = "/home/kasimov/notebooks/LegislatorImagePipeline/Senate_Embeddings"
FRAME_INTERVAL = 2
MIN_CLUSTER_SIZE = 50

def cosine_sim(a, b):
    a, b = np.ravel(a), np.ravel(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom != 0 else 0.0

def load_ground_truth(legislator_name: str):
    path = os.path.join(GROUND_TRUTH_BASE, f"{legislator_name}.npz")
    with np.load(path) as gt:
        if "centroid" in gt:
            return gt["centroid"]
        if "embedding" in gt:
            return gt["embedding"]
        return gt["arr_0"]

def run_clustering(legislator_name: str, video_path: str):
    # Use GPU 0 *within* the visible set (CUDA_VISIBLE_DEVICES is set by parent)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
    gc.collect()

    videoHandler = VideoHandler()
    faceExtractor = MTCNNFaceExtractor()
    featureExtractor = FacenetFeatureExtractor()

    # 1) Frames + Ground Truth
    X, _ = videoHandler.get_frames(video=video_path, interval=FRAME_INTERVAL)
    ground_truth_emb = load_ground_truth(legislator_name)

    # 2) Face Detection
    out = faceExtractor.batch_preprocess(X)
    if not out:
        return {"clusters": []}

    faces, boxes, _ = map(list, zip(*out))
    has_faces = [i for i, f in enumerate(faces) if len(f) > 0]
    if not has_faces:
        return {"clusters": []}

    faces = [faces[i] for i in has_faces]
    boxes = [boxes[i] for i in has_faces]

    # 3) Embeddings
    embeddings = featureExtractor.batch_extract_features(faces)
    all_embs_ = np.vstack([e.copy() for emb_list in embeddings for e in emb_list])
    all_boxes_ = np.array([b for frame_boxes in boxes for b in frame_boxes])

    # 4) Clustering
    clusterer = HDBScanClusterer(min_cluster_size=MIN_CLUSTER_SIZE)
    labels = clusterer.fit_predict(all_embs_)
    unique_labels = [l for l in np.unique(labels) if l != -1]

    results = {"clusters": []}
    for label in unique_labels:
        indices = np.where(labels == label)[0]
        cluster = all_embs_[indices]
        summary = clusterer.get_cluster_stats(cluster)

        medoid_idx = indices[summary["medoid_idx"]]
        cos_sims = [cosine_sim(ground_truth_emb, e) for e in cluster]

        results["clusters"].append({
            "n_points": int(summary["n_points"]),
            "mean_std": float(summary["mean_std"]),
            "sse": float(summary["sse"]),
            "density": float(len(indices) / len(all_embs_)),
            "cosine_sim_to_ground_truth": float(np.mean(cos_sims)),
            "medoid_box": all_boxes_[medoid_idx].tolist(),
            "centroid": summary["centroid"].tolist(),
        })

    return results

def main():
    legislator_name = sys.argv[1]
    video_path = sys.argv[2]
    results = run_clustering(legislator_name, video_path)

    sys.stdout.write(json.dumps(results))
    sys.stdout.flush()

if __name__ == "__main__":
    main()
