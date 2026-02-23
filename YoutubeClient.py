from pathlib import Path
import os
import google_auth_oauthlib.flow
import googleapiclient.discovery
import google.auth.transport.requests
import pickle
from yt_dlp import YoutubeDL

class YoutubeClient:
    def __init__(self, client_secret_path, creds_file="token.pickle", scopes=None):
        self.client_secret_path = client_secret_path
        self.creds_file = creds_file 
        self.scopes = ["https://www.googleapis.com/auth/youtube.readonly"] if scopes is None else scopes
        self.youtube = self._load_credentials_and_build_client()

    """
    Loads client secret (loads upon initialization)
    """
    def _load_credentials_and_build_client(self):
        os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
        creds = None
        # Load credentials if token.pickle exists
        if os.path.exists(self.creds_file):
            with open(self.creds_file, 'rb') as token:
                creds = pickle.load(token)
        # If there are no valid credentials, go through the OAuth flow
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(google.auth.transport.requests.Request())
            else:
                flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
                    self.client_secret_path, self.scopes)
                creds = flow.run_local_server(port=0)
            # Save credentials for next time
            with open(self.creds_file, 'wb') as token:
                pickle.dump(creds, token)
        # Build and return the YouTube API client
        youtube = googleapiclient.discovery.build(
            "youtube", "v3", credentials=creds)
        return youtube

    def search(self, **req_kwargs):
        request = self.youtube.search().list(
            **req_kwargs
        )
        response = request.execute()
        return response

    def exhaustive_search(self, max_total_results=None, **req_kwargs):
        results=[]
        next_page_token=None
        while True:
            response = self.search(
                **req_kwargs,
                pageToken=next_page_token
            )
            items = response.get("items", [])
            results.extend(items)
            next_page_token = response.get("nextPageToken")
            if not next_page_token or (max_total_results and len(results) >= max_total_results):
                break
        return results[:max_total_results]



    def download_video(self, video_id, output_dir="./", **yt_dlp_args):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ydl_opts = {
            "format_sort": ["res:720", "ext:mp4:m4a"],
            "postprocessors": [{
                "key": "FFmpegVideoConvertor",
                "preferedformat": "mp4",
            }],
            "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
            "noplaylist": True,
            "quiet": False,
            **yt_dlp_args
        }
        url = f"https://www.youtube.com/watch?v={video_id}" if len(video_id) == 11 else video_id
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return ydl.prepare_filename(info)


    def get_video_details(self, video_ids):
        """Retrieves details (like duration) for a list of video IDs."""
        if isinstance(video_ids, list):
            video_ids = ",".join(video_ids)
            
        request = self.youtube.videos().list(
            part="contentDetails,snippet,statistics", 
            id=video_ids
        )
        return request.execute()


    def get_video_details_exhaustive(self, video_ids):
        """Retrieves details for all provided video IDs, handling the 50-ID API limit."""
        if not video_ids:
            return []
        if isinstance(video_ids, str):
            video_ids = [video_ids]
        all_items = []
        for i in range(0, len(video_ids), 50):
            chunk = video_ids[i : i + 50]
            ids_string = ",".join(chunk)
            request = self.youtube.videos().list(
                part="contentDetails,snippet,statistics",
                id=ids_string
            )
            response = request.execute()
            items = response.get("items", [])
            all_items.extend(items)
        return all_items
