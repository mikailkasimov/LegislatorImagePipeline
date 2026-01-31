import requests

class GoogleSearchClient:
    def __init__(self, api_key, id):
        self.api_key = api_key
        self.id = id

    def image_search(self, query, num_pages=1,**kwargs):
        """
        Google image search
        
        param: query: str, search query
        param: num_pages: str, number of pages iterated through pagination
        
        return: (N, ) list of N links extracted from search results
        """
        all_links=[]
        url="https://www.googleapis.com/customsearch/v1"
        for start in range(1,num_pages*10+1,10):
            params = {
                "key": self.api_key,
                "cx": self.id,
                "q":query,
                "searchType": "image",
                "num": 10,
                "start": start
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            items = data.get("items", [])
            for item in items:
                all_links.append(item["link"])
        return all_links