from calendar import c
import datetime
from GoogleSearchClient import GoogleSearchClient
from FacenetFeatureExtractor import FacenetFeatureExtractor
from MTCNNFaceExtractor import MTCNNFaceExtractor
import os
import time
from pyfaktory import Client, Consumer, Job, Producer
import logging
import requests
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import threading
import shutil

load_dotenv()

GOOGLE_CUSTOM_SEARCH_ENGINE_API_KEY = os.getenv('GOOGLE_CUSTOM_SEARCH_ENGINE_API_KEY')
GOOGLE_CUSTOM_SEARCH_ENGINE_ID = os.getenv('GOOGLE_CUSTOM_SEARCH_ENGINE_ID')
GOOGLE_CLIENT_SECRET_PATH = os.getenv('GOOGLE_CLIENT_SECRET_PATH')
FAKTORY_URL = os.getenv('FAKTORY_URL')
BASE_DIR = "/home/kasimov/notebooks/LegislatorImagePipeline/Senate_Images"
EMBEDDDING_DIR = "/home/kasimov/notebooks/LegislatorImagePipeline/Senate_Embeddings"

logger = logging.getLogger("legislator image crawler")
logger.propagate = False
log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
numeric_level = getattr(logging, log_level_str, logging.INFO)
logger.setLevel(numeric_level)

if not logger.handlers:
    sh = logging.StreamHandler()
    fh = logging.FileHandler('legislator_image_crawler_log.txt') 
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    sh.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.addHandler(fh)

_tls = threading.local()

def get_search_client():
    if not hasattr(_tls, "search_client"):
        _tls.search_session = requests.Session()
        _tls.search_client = GoogleSearchClient(
            api_key=GOOGLE_CUSTOM_SEARCH_ENGINE_API_KEY,
            id=GOOGLE_CUSTOM_SEARCH_ENGINE_ID,
        )
    return _tls.search_client


def get_face_extractor():
    if not hasattr(_tls, "face_extractor"):
        _tls.face_extractor = MTCNNFaceExtractor()
    return _tls.face_extractor


def get_feature_extractor():
    if not hasattr(_tls, "feature_extractor"):
        _tls.feature_extractor = FacenetFeatureExtractor()
    return _tls.feature_extractor


def download_image_list(url_list: list[str], output_dir: str = "") -> None:
    logger.info("Entering: download_image_list()")
    logger.info(f"download_image_list(): output_dir={output_dir}, urls={len(url_list)}")
    os.makedirs(output_dir, exist_ok=True)
    headers = {
        "User-Agent": "LegislatorImagePipeline/1.0",
        "Accept": "image/*,*/*;q=0.8",
    }
    for i, url in enumerate(url_list):
        try:
            response = requests.get(url, timeout=15, headers=headers, stream=True)
            if response.status_code == 403:
                logger.info(f"download_image_list(): 403 Forbidden (skipping): {url}")
                continue
            response.raise_for_status()
            content_type = response.headers.get('content-type', '')
            if 'image' not in content_type:
                logger.info(f"download_image_list(): Skipping URL (not an image): {url} content-type={content_type}")
                continue
            extension = content_type.split('/')[-1].split(';')[0]
            if extension == 'jpeg':
                extension = 'jpg'
            filename = f"image_{i+1:03d}.{extension}"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        f.write(chunk)
            logger.info(f"download_image_list(): Successfully downloaded {filename}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error in download_image_list(): {e} url={url}")
            continue  # skip bad URLs instead of killing the whole job

    logger.info("Leaving: download_image_list()")


def process_folder(folder_path: str) -> tuple[float, np.ndarray | None]:
    logger.info("Entering: process_folder()")
    logger.info(f"process_folder(): folder_path={folder_path}")

    face_extractor = get_face_extractor()
    feature_extractor = get_feature_extractor()

    all_embeddings = []
    all_paths = []
    for fname in sorted(os.listdir(folder_path)):
        fpath = os.path.join(folder_path, fname)
        try:
            faces, _, _ = face_extractor.preprocess(fpath)
            logger.info(f"process_folder(): Extracted faces for {fpath} (shape {faces.shape})")
        except Exception as e:
            try:
                os.remove(fpath)
                logger.info(f"process_folder(): Deleted file after extract_faces error: {fpath}")
            except Exception as delete_err:
                logger.error(f"Error in process_folder(): Failed to delete {fpath}: {delete_err}")
            logger.error(f"Error in process_folder(): Error processing {fpath}: {e}")
            continue
        if int(faces.shape[0]) == 0:
            logger.info(f"process_folder(): No faces found in {fpath} (deleting)")
            try:
                os.remove(fpath)
            except Exception as delete_err:
                logger.error(f"Error in process_folder(): Failed to delete {fpath}: {delete_err}")
            continue
        elif int(faces.shape[0]) > 1:
            logger.info(f"process_folder(): More than 1 face found in {fpath} (deleting)")
            try:
                os.remove(fpath)
            except Exception as delete_err:
                logger.error(f"Error in process_folder(): Failed to delete {fpath}: {delete_err}")
            continue

        curr_embedding = feature_extractor.batch_extract_features(faces)
        assert len(curr_embedding) == 1
        # keep embeddings in consistent (1,512) shape (minimal but important)
        curr_embedding = np.asarray(curr_embedding[0]).reshape(1, -1)


        all_embeddings.append(curr_embedding)
        all_paths.append(fpath)
        logger.info(f"process_folder(): Added embedding for {fpath}. total_embeddings={len(all_embeddings)}")

    if len(all_embeddings) == 0:
        logger.info("process_folder(): No valid embeddings collected; returning (0.0, None)")
        logger.info("Leaving: process_folder()")
        return 0.0, None

    duplicate_face_paths = []
    duplicate_face_embeddings_indices = set()
    sets = set()
    for i in range(0, len(all_embeddings)):
        for j in range(0, len(all_embeddings)):
            if (i, j) in sets or (j, i) in sets or i == j or i in duplicate_face_embeddings_indices or j in duplicate_face_embeddings_indices:
                continue
            sets.add((i, j))
            sets.add((j, i))

            v1 = all_embeddings[i]
            v2 = all_embeddings[j]
            assert len(v1) == len(v2)
            similarity = cosine_similarity(v1, v2).flatten()
            if (similarity) > 0.90:
                logger.info(f"process_folder(): DUPLICATE found: {all_paths[i]} similarity={similarity} with {all_paths[j]}")
                duplicate_face_paths.append(all_paths[j])
                duplicate_face_embeddings_indices.add(j)

    new_all_embeddings = [x for i, x in enumerate(all_embeddings) if i not in duplicate_face_embeddings_indices]
    new_all_paths = [x for i, x in enumerate(all_paths) if x not in duplicate_face_paths]
    outlier_face_embeddings_indices = set()
    centroid = np.mean(new_all_embeddings, axis=0, keepdims=False)

    logger.info(f"process_folder(): After duplicate removal: remaining={len(new_all_paths)} removed={len(all_paths)-len(new_all_paths)}")

    for i, v2 in enumerate(new_all_embeddings):
        similarity = cosine_similarity(centroid, v2).flatten()
        if similarity < 0.6:
            logger.info(f"process_folder(): OUTLIER found: {new_all_paths[i]} score={similarity}")
            outlier_face_embeddings_indices.add(i)

    final_all_embeddings = [x for i, x in enumerate(new_all_embeddings) if i not in outlier_face_embeddings_indices]
    final_all_paths = [x for i, x in enumerate(new_all_paths) if i not in outlier_face_embeddings_indices]

    delete_set = set(all_paths) - set(final_all_paths)
    n = len(final_all_paths)
    d = 512
    logger.info(f"process_folder(): After outlier removal: remaining={len(final_all_paths)} removed={len(delete_set)}")
    for path in delete_set:
        try:
            os.remove(path)
            logger.info(f"process_folder(): Deleted {path}")
        except Exception as delete_err:
            logger.error(f"Error in process_folder(): Failed to delete {path}: {delete_err}")

    if len(final_all_embeddings) == 0:
        logger.info("process_folder(): No embeddings left after filtering")
        logger.info("Leaving: process_folder()")
        return 0.0, None

    final_centroid = np.mean(final_all_embeddings, axis=0, keepdims=False)
    final_all_embeddings = np.array(final_all_embeddings).reshape(n, d)
    avg_cosine_sim = cosine_similarity(final_centroid, final_all_embeddings).mean()

    logger.info(f"process_folder(): avg_cosine_sim={avg_cosine_sim}")
    logger.info("Leaving: process_folder()")
    return avg_cosine_sim, final_centroid


def enqueue_search_legislator(name):
    logger.info("Entering: enqueue_search_legislator()")
    logger.info(f"enqueue_search_legislator(): name={name}")

    output_dir = os.path.join(BASE_DIR, name)
    if os.path.exists(output_dir):
        logger.info(f"enqueue_search_legislator(): Skipping (already exists): {output_dir}")
        logger.info("Leaving: enqueue_search_legislator()")
        return

    chamber = name.split('_')[0]
    name_arr = name.split('_')[1:]
    pretty_name = " ".join(name_arr)
    query = f"{chamber}. {pretty_name} headshot"
    logger.info(f"enqueue_search_legislator(): query={query}")

    search_client = get_search_client()
    try:
        links = search_client.image_search(query,num_pages=5)
        logger.info(f"enqueue_search_legislator(): Found {len(links)} image links")

        download_image_list(links, output_dir)

        with Client(faktory_url=FAKTORY_URL, role="producer") as c:
            producer = Producer(client=c)
            job = Job(
                jobtype="process_folder",
                args=(output_dir,),
                queue="process_folder"
            )
            producer.push(job)

        logger.info(f"enqueue_search_legislator(): Enqueued process_folder job for {output_dir}")
        logger.info("Leaving: enqueue_search_legislator()")

    except Exception as e:
        logger.error(f"Error in enqueue_search_legislator(): {e}")
        try:
            shutil.rmtree(output_dir, ignore_errors=True)
            logger.info(f"enqueue_search_legislator(): Cleaned up {output_dir}")
        except Exception as cleanup_err:
            logger.error(f"Error in enqueue_search_legislator(): Cleanup failed for {output_dir}: {cleanup_err}")
        raise


def enqueue_process_folder(folder_path):
    logger.info("Entering: enqueue_process_folder()")
    logger.info(f"enqueue_process_folder(): folder_path={folder_path}")
    try:
        os.makedirs(EMBEDDDING_DIR, exist_ok=True)
        avg_cosine_sim, centroid = process_folder(folder_path)
        centroid_name = os.path.basename(folder_path.rstrip("/"))

        try:
            n_after = len([
                f for f in os.listdir(folder_path)
                if os.path.isfile(os.path.join(folder_path, f))
            ])
        except Exception as e:
            logger.error(f"Error in enqueue_process_folder(): Failed counting images in {folder_path}: {e}")
            n_after = 0

        if centroid is None:
            logger.info(f"enqueue_process_folder(): No centroid produced for {centroid_name}")
            logger.info("Leaving: enqueue_process_folder()")
            return

        centroid_path = os.path.join(EMBEDDDING_DIR, f"{centroid_name}.npz")
        np.savez(
            centroid_path,
            centroid=np.asarray(centroid),
            name=centroid_name,
            num_images=n_after,
            avg_cos=float(avg_cosine_sim),
            centroid_path=centroid_path,
        )
        logger.info(f"enqueue_process_folder(): Saved embedding npz to {centroid_path}")
        logger.info("Leaving: enqueue_process_folder()")
    except Exception as e:
        logger.error(f"Error in enqueue_process_folder(): {e}")
        raise


if __name__ == "__main__":
    logger.info("Starting Faktory consumer...")
    with Client(faktory_url=FAKTORY_URL, role="consumer") as c:
        consumer = Consumer(
            client=c,
            queues=["default", "search_legislator", "process_folder"],
            concurrency=5
        )
        consumer.register("search_legislator", enqueue_search_legislator)
        consumer.register("process_folder", enqueue_process_folder)
        consumer.run()
        logger.info("Listening for tasks...")
