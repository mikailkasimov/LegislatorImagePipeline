from pathlib import Path
import os
import google_auth_oauthlib.flow
import googleapiclient.discovery
import google.auth.transport.requests
import pickle
from yt_dlp import YoutubeDL
from typing import Any, Dict, List, Optional, Union

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

    def search(self, **req_kwargs: Any) -> Dict[str, Any]:
        """
        Performs a single basic search using the YouTube Data API
        Ref: https://developers.google.com/youtube/v3/docs/search/list

        param: req_kwargs: Standard YouTube API parameters (q, part, type, etc.)
        return: response: Dict containing 'items' list and pagination metadata
        """
        request = self.youtube.search().list(**req_kwargs)
        return request.execute()

    def exhaustive_search(self, max_total_results: Optional[int] = None, **req_kwargs: Any) -> List[Dict[str, Any]]:
        """
        Paginates through search results to bypass the 50-item API limit
        Ref: https://developers.google.com/youtube/v3/guides/implementation/pagination

        param: max_total_results: Total number of items to return (None for all)
        param: req_kwargs: Standard search parameters (q, part, etc.)
        return: results: List of search result resource objects
        """
        results: List[Dict[str, Any]] = []
        next_page_token: Optional[str] = None
        
        while True:
            response = self.search(**req_kwargs, pageToken=next_page_token)
            items = response.get("items", [])
            results.extend(items)
            
            next_page_token = response.get("nextPageToken")
            if not next_page_token or (max_total_results and len(results) >= max_total_results):
                break
        return results[:max_total_results]


    def download_video(self, video_id: str, output_dir: str = "./", **yt_dlp_args: Any) -> str:
        """
        Downloads a YouTube video using yt-dlp with optimized MP4 settings
        Ref: https://github.com/yt-dlp/yt-dlp

        param: video_id: 11-character YouTube video ID
        param: output_dir: Local directory where the file should be saved
        param: yt_dlp_args: Overrides for the YoutubeDL options dictionary
        return: filepath: Absolute file path of the downloaded video
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        ydl_opts = {
            "format_sort": ["res:720", "ext:mp4:m4a"],
            "postprocessors": [{
                "key": "FFmpegVideoConvertor",
                "preferedformat": "mp4",
            }],
            "outtmpl": str(out_path / "%(id)s.%(ext)s"),
            "noplaylist": True,
            "quiet": False,
            **yt_dlp_args
        }
        url = f"https://www.youtube.com/watch?v={video_id}"
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return str(ydl.prepare_filename(info))

    def get_video_details(self, video_ids: Union[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Retrieves metadata (duration, stats, tags) for a list of video IDs
        Ref: https://developers.google.com/youtube/v3/docs/videos/list

        param: video_ids: A single video ID string or a list of ID strings
        return: items: List of video resource objects (contentDetails, statistics)
        """
        if not video_ids:
            return []
        if isinstance(video_ids, str):
            video_ids = [video_ids]
            
        all_items: List[Dict[str, Any]] = []
        for i in range(0, len(video_ids), 50):
            chunk = ",".join(video_ids[i : i + 50])
            response = self.youtube.videos().list(
                part="contentDetails,snippet,statistics",
                id=chunk
            ).execute()
            all_items.extend(response.get("items", []))
            
        return all_items