import sys
import isodate
import asyncio
from datetime import timedelta
from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.worker import Worker
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions
import os
import subprocess
import json
import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import Json
from psycopg2.extensions import register_adapter
import contextvars
import threading
import time
from functools import wraps
from typing import Any, Callable, TypeVar, cast
import concurrent.futures
import argparse
from YoutubeClient import YoutubeClient
from SambaClient import SambaClient
from datetime import datetime
import selectors

register_adapter(dict, Json)
load_dotenv()

# ---- environment variables ----
GOOGLE_CLIENT_SECRET_PATH = os.getenv("GOOGLE_CLIENT_SECRET_PATH")
FAKTORY_URL = os.getenv('FAKTORY_URL')
DATABASE_URL = os.getenv('DATABASE_URL')
SERVER = "skanda.cs.binghamton.edu"
SHARE  = "podcastdb"
USER   = os.getenv("SAMBA_USER")
PASS   = os.getenv("SAMBA_PASS")

BASE_PATH = "/app"
ANACONDA_BASE = "/anaconda"

FACENET_PYTHON_BIN = "/opt/conda/envs/env_facenet/bin/python"
WHISPER_PYTHON_BIN = "/opt/conda/envs/env_whisper/bin/python"
# WHISPER_PYTHON_BIN = f"{ANACONDA_BASE}/envs/whisper-diarization/bin/python"
# FACENET_PYTHON_BIN = f"{ANACONDA_BASE}/envs/facenet/bin/python"

DIARIZE_PATH = f"{BASE_PATH}/whisper-diarization/diarize.py"
GET_CLUSTERS_PATH = f"{BASE_PATH}/run_get_clusters.py"
COOKIE_FILE = f"{BASE_PATH}/cookies.txt"
TEMP_VIDS_BASE = f"{BASE_PATH}/tmp_vids/"


# ----- helpers ------
def run_gpu_subprocess(cmd_args, prefix="gpu-task"):
    """Generic helper to run subprocesses with live logging and JSON output."""
    p = subprocess.Popen(
        cmd_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    sel = selectors.DefaultSelector()
    sel.register(p.stdout, selectors.EVENT_READ, "stdout")
    sel.register(p.stderr, selectors.EVENT_READ, "stderr")
    out, err = [], []
    while True:
        for key, _ in sel.select(timeout=0.2):
            line = key.fileobj.readline()
            if not line:
                sel.unregister(key.fileobj)
                continue
            if key.data == "stderr":
                err.append(line)
                print(f"[{prefix}-err] {line}", end="") # Live streaming
            else:
                out.append(line)
        if p.poll() is not None and not sel.get_map():
            break
    rc = p.wait()
    stdout_text = "".join(out).strip()
    stderr_text = "".join(err)
    if rc != 0 or not stdout_text or stdout_text[0] not in "[{":
        error_msg = f"{prefix} failed rc={rc}. "
        if not stdout_text:
            error_msg += "No stdout produced."
        elif stdout_text[0] not in "[{":
            error_msg += f"Invalid JSON format: {stdout_text[:100]}..."
        print(f"ERROR: {error_msg}\nSTDERR: {stderr_text[-1000:]}")
        raise RuntimeError(error_msg)
    return json.loads(stdout_text)

def video_is_downloaded(legislator_name, video_id):
    smb_client = SambaClient(server=SERVER, share=SHARE, user=USER, password=PASS)
    return smb_client.exists(f"./db_new/{legislator_name}/{video_id}.mp4")

F = TypeVar("F", bound=Callable[..., Any])
MIN_HEARTBEAT_INTERVAL = 10.0
def sync_auto_heartbeater(fn: F) -> F:
    """Decorator to automatically heartbeat a sync activity while it is running."""
    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        info = activity.info()
        heartbeat_timeout = info.heartbeat_timeout
        stop_event = threading.Event()
        heartbeat_thread: threading.Thread | None = None
        if heartbeat_timeout:
            # Copy current activity context so heartbeat works in new thread
            ctx = contextvars.copy_context()
            # Heartbeat twice as often as timeout, but at least every MIN_HEARTBEAT_INTERVAL
            delay = min(
                MIN_HEARTBEAT_INTERVAL,
                heartbeat_timeout.total_seconds() / 2,
            )
            def heartbeat_loop() -> None:
                while not stop_event.is_set():
                    time.sleep(delay)
                    # This runs inside the copied activity context
                    activity.heartbeat(
                        f"Heartbeating at {datetime.now()} "
                        f"for {info.activity_type!r} id={info.activity_id!r}"
                    )
            heartbeat_thread = threading.Thread(
                target=ctx.run,
                args=(heartbeat_loop,),
                daemon=True,
            )
            heartbeat_thread.start()
        try:
            return fn(*args, **kwargs)
        finally:
            if heartbeat_thread:
                stop_event.set()
                heartbeat_thread.join()
    return cast(F, wrapper)


# --- 1. ACTIVITIES ---

class YoutubeActivities:
    from YoutubeClient import YoutubeClient
    def __init__(self, yt_client: YoutubeClient) -> None:
        self.yt_client = yt_client
    @activity.defn
    def search_legislator(self, query: str, max_results: int):
        yt_results = self.yt_client.exhaustive_search(
            max_total_results=max_results,
            part="snippet",
            q=query,
            type="video",
            videoDuration="long",
            maxResults=50,
            topicId="/m/05qt0",
            order="viewCount",
        )
        all_ids = [res["id"]["videoId"] for res in yt_results]
        return all_ids

    @activity.defn
    def get_video_details(self, all_ids: list[str]):
        details = self.yt_client.get_video_details(all_ids)
        return details

    @activity.defn
    def download_video(self, video_id: str, output_dir: str):
        path = self.yt_client.download_video(
            video_id=video_id,
            output_dir=output_dir,  #TEMP_VIDS_BASE,
            cookiefile=COOKIE_FILE,
            remote_components=["ejs:github"],
            player_client=["web", "tv"],
        )
        return path


#synchronous because gpu-heavy task
@activity.defn
@sync_auto_heartbeater
def transcribe_video(video_path: str):
    """Whisper Diarization Activity."""
    cmd = [WHISPER_PYTHON_BIN, "-u", DIARIZE_PATH, "-a", video_path]
    return run_gpu_subprocess(cmd, prefix="whisper")

#synchronous because gpu-heavy task
@activity.defn
@sync_auto_heartbeater
def get_clusters(legislator_name: str, video_path: str):
    """Facenet Clustering Activity."""
    cmd = [FACENET_PYTHON_BIN, "-u", GET_CLUSTERS_PATH, legislator_name, video_path]
    return run_gpu_subprocess(cmd, prefix="facenet")


@activity.defn
def save_metadeta(legislator_name, cluster_data, srt_data, yt_data, video_path):
    #maybe move this logic elsewhere
    legislator_arr = legislator_name.split('_')
    chamber = legislator_arr[0]
    display_name = ' '.join(legislator_arr[1:])
    
    base_path = os.path.splitext(video_path)[0]
    video_id = os.path.basename(base_path)

    conn = psycopg2.connect(dsn=DATABASE_URL)
    cur = conn.cursor()

    q = """
        INSERT INTO legislator_video_data 
        (legislator_name, video_id, chamber, name, cluster_info, srt_data, yt_metadata, processed_at)
        VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, NOW())
        ON CONFLICT (legislator_name, video_id) 
        DO UPDATE SET 
            cluster_info = EXCLUDED.cluster_info, 
            srt_data = EXCLUDED.srt_data, 
            yt_metadata = EXCLUDED.yt_metadata,
            processed_at = NOW();
    """
    cur.execute(q, (legislator_name, video_id, chamber, display_name, Json(cluster_data), Json(srt_data), Json(yt_data)))
    conn.commit()
    cur.close()
    conn.close()


@activity.defn
def clean_up(legislator_name, video_path, video_id):
    smb_client = SambaClient(server=SERVER, share=SHARE, user=USER, password=PASS)
    remote_video_path = f"db_new/{legislator_name}/{video_id}.mp4"
    if not video_is_downloaded(legislator_name, video_id):
        smb_client.write_file(local_path=video_path, remote_path=remote_video_path)

    if os.path.exists(video_path):
        os.remove(video_path)
    

# --- 2. WORKFLOW ---

@workflow.defn(sandboxed=False)
class LegislatorOrchestratorWorkflow:
    @workflow.run
    async def run(self, legislator_name: str, query: str, max_results: int) -> None:
        ids = await workflow.execute_activity(
            YoutubeActivities.search_legislator,
            args=[query, max_results],
            start_to_close_timeout=timedelta(minutes=5),
        )
        details = await workflow.execute_activity(
            YoutubeActivities.get_video_details,
            args=[ids],
            start_to_close_timeout=timedelta(minutes=5),
        )
        # Filter by duration (< 4 hours)
        filtered_details = [
            item
            for item in details
            if isodate.parse_duration(
                item["contentDetails"]["duration"]
            ).total_seconds()
            < 14400
        ]
        child_tasks = []
        for info_json in filtered_details:
            task = workflow.start_child_workflow(
                VideoPipelineWorkflow.run,
                args=[legislator_name, info_json],
                id=f"{legislator_name}-{info_json['id']}",
            )
            child_tasks.append(task)
        await asyncio.gather(*child_tasks)


@workflow.defn(sandboxed=False)
class VideoPipelineWorkflow:
    @workflow.run
    async def run(self, legislator_name: str, info_json: dict):
        video_id = info_json['id']
        video_path = await workflow.execute_activity(
            YoutubeActivities.download_video,
            args=[video_id, TEMP_VIDS_BASE],
            task_queue="legislator-io",
            start_to_close_timeout=timedelta(minutes=60),
        )
        srt_data = await workflow.execute_activity(
            transcribe_video,
            args=[video_path],
            task_queue="legislator-gpu",
            start_to_close_timeout=timedelta(hours=2),
        )
        cluster_data = await workflow.execute_activity(
            get_clusters,
            args=[legislator_name, video_path],
            task_queue="legislator-gpu",
            start_to_close_timeout=timedelta(hours=2),
        )
        await workflow.execute_activity(
            save_metadeta,
            args=[legislator_name, cluster_data, srt_data, info_json, video_path],
            task_queue="legislator-io",
            start_to_close_timeout=timedelta(minutes=5),
        )
        await workflow.execute_activity(
            clean_up,
            args=[legislator_name, video_path, video_id],
            task_queue="legislator-io",
            start_to_close_timeout=timedelta(minutes=5),
        )

# --- 3. RUNNER ---


async def main():
    parser = argparse.ArgumentParser(description="Run Temporal Workers")
    parser.add_argument("-q", "--queue", type=str, required=True, 
                        help="Specify worker type: 'io' or 'gpu'")
    args = parser.parse_args()

    temporal_addr = os.getenv("TEMPORAL_ADDRESS", "temporal:7233")
    client = await Client.connect(temporal_addr)
    
    yt_client = YoutubeClient(client_secret_path=GOOGLE_CLIENT_SECRET_PATH)
    ytact = YoutubeActivities(yt_client)

    if args.queue == "io":
        activity_executor = concurrent.futures.ThreadPoolExecutor(max_workers=100)
        try:
            worker = Worker(
                client,
                task_queue="legislator-io",
                workflows=[LegislatorOrchestratorWorkflow, VideoPipelineWorkflow],
                activities=[
                    ytact.search_legislator, 
                    ytact.get_video_details, 
                    ytact.download_video, 
                    save_metadeta, 
                    clean_up,
                ],
                activity_executor=activity_executor,
                max_concurrent_activities=20,
            )
            print(f"Worker listening on queue: {worker.task_queue}")
            await worker.run()
        except Exception as e:
            print(f"Worker crashed: {e}")
        finally:
            print("Shutting down activity executor...")
            activity_executor.shutdown(wait=True)

    elif args.queue == "gpu":
        activity_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        try:
            worker = Worker(
                client,
                task_queue="legislator-gpu",
                workflows=[LegislatorOrchestratorWorkflow, VideoPipelineWorkflow],
                activities=[transcribe_video, get_clusters],
                activity_executor=activity_executor,
                max_concurrent_activities=1,
            )
            print(f"Worker listening on queue: {worker.task_queue}")
            await worker.run()
        except Exception as e:
            print(f"Worker crashed: {e}")
        finally:
            print("Shutting down activity executor...")
            activity_executor.shutdown(wait=True)
    else:
        print(f"Error: Invalid queue type '{args.queue}'. Must be 'io' or 'gpu'.")
        sys.exit(1)

    print(f"Worker listening on queue: {worker.task_queue}")
    await worker.run()

if __name__ == "__main__":
    asyncio.run(main())