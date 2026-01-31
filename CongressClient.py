from typing import Any
import requests

class CongressClient:
    def __init__(self, api_key):
        self.api_key = api_key

    def get_current_congress(self) -> list[dict[str, Any]]:
            """ 
            Get current congress

            return: (N,) list of N congress members
            """
            all_members = []
            url = "https://api.congress.gov/v3/member?"
            for i in range(0,750, 250):
                args = {
                    "api_key": self.api_key,
                    "currentMember": True,
                    "limit": 250,
                    "offset": i
                }
                response= requests.get(url, params=args)
                data = response.json()
                members = data["members"]
                for member in members:
                    all_members.append(member)
            return all_members