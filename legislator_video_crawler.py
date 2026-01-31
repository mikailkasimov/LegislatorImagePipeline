import isodate
import json
import sys
import re
import torch
import gc
import os
import time
import subprocess
import logging
import numpy as np
import psycopg2
import datetime
from dotenv import load_dotenv
from pyfaktory import Client, Consumer, Job, Producer
from psycopg2.extras import Json
from psycopg2.extensions import register_adapter

# Custom Client Imports
from SambaClient import SambaClient
from YoutubeClient import YoutubeClient
from MTCNNFaceExtractor import MTCNNFaceExtractor
from FacenetFeatureExtractor import FacenetFeatureExtractor
from VideoHandler import VideoHandler

# Allow psycopg2 to insert dicts into JSONB columns
register_adapter(dict, Json)
load_dotenv()

# Environment Variables
GOOGLE_CLIENT_SECRET_PATH = os.getenv("GOOGLE_CLIENT_SECRET_PATH")
FAKTORY_URL = os.getenv('FAKTORY_URL')
DATABASE_URL = os.getenv('DATABASE_URL')
SERVER = "skanda.cs.binghamton.edu"
SHARE  = "podcastdb"
USER   = os.getenv("SAMBA_USER")
PASS   = os.getenv("SAMBA_PASS")
BASE_PATH = "/home/kasimov/notebooks/LegislatorImagePipeline"
# Logging Implementation
logger = logging.getLogger("legislator video crawler")
logger.propagate = False
log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
numeric_level = getattr(logging, log_level_str, logging.INFO)
logger.setLevel(numeric_level)

if not logger.handlers:
    sh = logging.StreamHandler()
    fh = logging.FileHandler('legislator_video_crawler_log.txt') 
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    sh.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.addHandler(fh)

# Global Clients
yt_client = YoutubeClient(client_secret_path=GOOGLE_CLIENT_SECRET_PATH)
videoHandler = VideoHandler()
featureExtractor = FacenetFeatureExtractor()
faceExtractor = MTCNNFaceExtractor()

ground_truth_embedding_base = f"{BASE_PATH}/Senate_Embeddings"

# --- Helper Functions ---
def video_id_is_processed(video_id):
    query = "SELECT EXISTS(SELECT 1 FROM legislator_video_data WHERE video_id = %s);"
    try:
        with psycopg2.connect(dsn=DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(query, (video_id,))
                exists = cur.fetchone()[0]
                return exists
    except Exception as e:
        logger.error(f"Database error checking video_id {video_id}: {e}")
        return False
    
def video_is_downloaded(legislator_name, video_id):
    smb_client = SambaClient(server=SERVER, share=SHARE, user=USER, password=PASS)
    return smb_client.exists(f"./db_new/{legislator_name}/{video_id}.mp4")

def parse_diarized_srt(file_path):
    logger.info(f"Entering: parse_diarized_srt({file_path})")
    try:
        if not os.path.exists(file_path):
            logger.warning(f"SRT file not found: {file_path}")
            return []
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(Speaker \d+): (.*)'
        matches = re.findall(pattern, content)
        result = [
            {"index": int(m[0]), "start": m[1], "end": m[2], "speaker": m[3], "text": m[4].strip()}
            for m in matches
        ]
        logger.info(f"Leaving: parse_diarized_srt() - Extracted {len(result)} lines")
        return result
    except Exception as e:
        logger.error(f"Error in parse_diarized_srt(): {e}")
        raise

# --- Task Enqueue Functions ---

def enqueue_search_yt(legislator_name, query, max_results):
    logger.info(f"Entering: enqueue_search_yt() for {legislator_name}")
    try:
        yt_results = yt_client.exhaustive_search(
            max_total_results=max_results, part="snippet", q=query, type="video",
            videoDuration="long", maxResults=50, topicId="/m/05qt0", order="viewCount"
        )
        all_ids = [res['id']['videoId'] for res in yt_results]  #video_id
        
        details = yt_client.get_video_details_exhaustive(all_ids)   #yt_metadata
        
        #queue next step (download videos)
        with Client(faktory_url=FAKTORY_URL, role="producer") as c:
            producer = Producer(client=c)
            for item in details:
                duration_str = item['contentDetails']['duration']   #duration of video
                duration_s = isodate.parse_duration(duration_str).total_seconds()
                if duration_s < 14400:  #we filter out videos longer than 4 hours
                    vid_id = item['id']
                    job = Job(jobtype="download_video", args=(legislator_name, vid_id, item), 
                              queue="download_video", custom={"priority":1})
                    producer.push(job)
        logger.info(f"Leaving: enqueue_search_yt()")
    except Exception as e:
        logger.error(f"Error in enqueue_search_yt(): {e}")
        raise

def enqueue_download_video(legislator_name, video_id, yt_metadata):
    logger.info(f"Entering: enqueue_download_video() for ID {video_id}")
    local_path=f"{BASE_PATH}/{video_id}.mp4"
    #video is already in db, we do not consider it
    if video_id_is_processed(video_id):
        logger.info(f"Video {video_id} already exists and processed, skipping!")
        return
    #video is already downloaded, so we pull from the server
    if video_is_downloaded(legislator_name, video_id):
        logger.info(f"Video {video_id} exists in the server, pulling from server!")
        smb_client = SambaClient(server=SERVER, share=SHARE, user=USER, password=PASS)
        smb_client.read_file(
            local_path=f"{BASE_PATH}/{video_id}.mp4",
            remote_path=f"./db_new/{legislator_name}/{video_id}.mp4"
        )
    #else, we download the video
    else:
        try:
            yt_client.download_video(
                video_id,
                cookiefile=f"{BASE_PATH}/cookies.txt",          #using cookies from account logged in on youtube (   mkbingtest1@gmail.com   )
                remote_components=["ejs:github"],       #for deno JS runtime machine
                player_client=["web", "tv"] 
            )
        except Exception as e:
            logger.error(f"Error in enqueue_download_video(): {e}")
            raise
    #queue next step (transcription)
    with Client(faktory_url=FAKTORY_URL, role="producer") as c:
        producer = Producer(client=c)
        job = Job(jobtype="transcribe_video", args=(legislator_name, local_path, yt_metadata), 
                    queue="transcribe_video", custom={"priority":3}, retry=5)
        producer.push(job)
    logger.info(f"Leaving: enqueue_download_video()")

def enqueue_transcribe_video(legislator_name, video_path, yt_metadata):
    logger.info(f"Entering: enqueue_transcribe_video() - Path: {video_path}")
    diarize_path = f"{BASE_PATH}/whisper-diarization/diarize.py"
    python_bin = "/home/kasimov/anaconda3/envs/whisper-diarization/bin/python"
    gpu_pool = ["0", "1", "2", "3"]
    success = False

    #try each gpu available 
    for gpu_id in gpu_pool:
        try:
            logger.info(f"Attempting transcription on GPU {gpu_id} for {video_path}")
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = gpu_id
            subprocess.run([python_bin, diarize_path, "-a", video_path], env=env, check=True, capture_output=False, timeout=10800)
            success = True
            break
        except subprocess.CalledProcessError as e:
            logger.warning(f"GPU {gpu_id} failed with exit code {e.returncode} for {video_path}. Trying next...")

    if not success:
        logger.error(f"FATAL: All GPUs failed for {video_path}")
        raise RuntimeError(f"All GPUs exhausted for {video_path}")

    # queue next step (cluster information of faces)
    try:
        with Client(faktory_url=FAKTORY_URL, role="producer") as c:
            producer = Producer(client=c)
            job = Job(jobtype="get_clusters", args=(legislator_name, video_path, yt_metadata), 
                      queue="get_clusters", custom={"priority":5}, retry=5)
            producer.push(job)
        logger.info("Leaving: enqueue_transcribe_video()")
    except Exception as e:
        logger.error(f"Queue Error in transcribe_video for {video_path}: {e}")
        raise

def enqueue_get_clusters(legislator_name, video_path, yt_metadata):
    logger.info(f"Entering: enqueue_get_clusters() - {legislator_name}")
    python_bin = sys.executable
    worker_script = f"{BASE_PATH}/run_get_clusters.py"
    gpu_pool = ["0", "1", "2", "3"]
    last_err = None
    for gpu_id in gpu_pool:
        try:
            logger.info(f"Attempting clustering via subprocess on GPU {gpu_id} for {video_path}")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            env["PYTHONUNBUFFERED"] = "1"
            p = subprocess.run(
                [python_bin, worker_script, legislator_name, video_path],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                timeout=10800,
            )
            results = json.loads(p.stdout)

            # queue next step (save all metadata to postgres) 
            with Client(faktory_url=FAKTORY_URL, role="producer") as c:
                producer = Producer(client=c)
                job = Job(jobtype="save_metadata", args=(legislator_name, results, video_path, yt_metadata), 
                          queue="save_metadata", custom={"priority":7})
                producer.push(job)

            logger.info(f"Leaving: enqueue_get_clusters() - Success on GPU {gpu_id}")
            return
        except subprocess.CalledProcessError as e:
            last_err = e.stderr[-500:] if e.stderr else str(e)
            logger.warning(f"GPU {gpu_id} failed (exit={e.returncode}). stderr tail:\n{last_err}")
            continue
        except json.JSONDecodeError as e:
            last_err = f"JSON decode failed. stdout tail:\n{p.stdout[-500:]}"
            logger.error(last_err)
            raise
    raise RuntimeError(f"All GPUs exhausted for clustering {video_path}. Last error:\n{last_err}")

def enqueue_save_metadata(legislator_name, results_json, video_path, yt_metadata):
    logger.info(f"Entering: enqueue_save_metadata() for {legislator_name}")
    try:
        legislator_arr = legislator_name.split('_')
        chamber = legislator_arr[0]
        display_name = ' '.join(legislator_arr[1:])
        
        base_path = os.path.splitext(video_path)[0]
        video_id = os.path.basename(base_path)
        srt_json = parse_diarized_srt(f"{base_path}.srt")

        conn = psycopg2.connect(dsn=DATABASE_URL)
        cur = conn.cursor()

        q = """
            INSERT INTO legislator_video_data 
            (legislator_name, video_id, chamber, name, cluster_info, srt_data, yt_metadata, processed_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (legislator_name, video_id) 
            DO UPDATE SET 
                cluster_info = EXCLUDED.cluster_info, 
                srt_data = EXCLUDED.srt_data, 
                yt_metadata = EXCLUDED.yt_metadata,
                processed_at = NOW();
        """
        cur.execute(q, (legislator_name, video_id, chamber, display_name, Json(results_json), Json(srt_json), Json(yt_metadata)))
        conn.commit()
        cur.close()
        conn.close()

        # queue final step (clean up any temporary files made)
        with Client(faktory_url=FAKTORY_URL, role="producer") as c:
            producer = Producer(client=c)
            job = Job(jobtype="clean_up", args=(legislator_name, video_path), 
                      queue="clean_up", custom={"priority":9})
            producer.push(job)
        logger.info(f"Leaving: enqueue_save_metadata() - Saved data for {video_id}")
    except Exception as e:
        logger.error(f"Error in enqueue_save_metadata(): {e}")
        raise

def enqueue_clean_up(legislator_name, video_path):
    logger.info(f"Entering: enqueue_clean_up() - {video_path}")
    base_path = os.path.splitext(video_path)[0]
    video_id = os.path.basename(base_path)
    remote_video_path = f"db_new/{legislator_name}/{video_id}.mp4"
    
    try:
        smb_client = SambaClient(server=SERVER, share=SHARE, user=USER, password=PASS)
        if not video_is_downloaded(legislator_name, video_id):
            smb_client.write_file(local_path=video_path, remote_path=remote_video_path)
        
        if os.path.exists(video_path):
            os.remove(video_path)
            
        for ext in ['.srt', '.txt']:
            file_to_del = f"{base_path}{ext}"
            if os.path.exists(file_to_del):
                os.remove(file_to_del)
        
        logger.info(f"Leaving: enqueue_clean_up() - Successfully moved {video_id} to SMB and cleaned local storage")
    except Exception as e:
        logger.error(f"Error in enqueue_clean_up(): {e}")
        raise


# --- Consumer Entrypoint ---

if __name__ == "__main__":
    logger.info("Starting Faktory consumer...")
    with Client(faktory_url=FAKTORY_URL, role="consumer") as c:
        consumer = Consumer(
            client=c,
            queues=["clean_up", "save_metadata", "get_clusters", "transcribe_video", "download_video", "search_legislator"],
            concurrency=15,
            priority='strict'
        )
        consumer.register("search_legislator", enqueue_search_yt)
        consumer.register("download_video", enqueue_download_video)
        consumer.register("transcribe_video", enqueue_transcribe_video)
        consumer.register("get_clusters", enqueue_get_clusters)
        consumer.register("save_metadata", enqueue_save_metadata)
        consumer.register("clean_up", enqueue_clean_up)
        
        logger.info("Listening for tasks...")
        consumer.run()