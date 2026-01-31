import time
import logging
from pyfaktory import Client, Consumer, Job, Producer
import sys
from CongressClient import CongressClient 
import os
from dotenv import load_dotenv

load_dotenv()

CONGRESS_API_KEY = os.getenv('CONGRESS_API_KEY')
FAKTORY_URL = os.getenv('FAKTORY_URL')

def congress_to_title_names(member_list: list[dict]) -> list[str]:
    """ 
    Extracts title and name from list(dict) produced by CongressClient.get_current_congress()

    param: member_list: list(dict), containing a json for each member
    return: url_list: list(str), containing sorted list of "title_name" for each congress member
    """
    try:
        url_list=[]
        for i,member in enumerate(member_list):
            name = member.get('name')
            name_arr = name.split(', ')
            name_arr = [n.replace(' ', '_').replace('.', '') for n in name_arr]
            name_arr.reverse()
            formatted_name = '_'.join(name_arr)
            chamber_info = member.get('terms', {}).get('item', [])[0].get('chamber')
            chamber = 'Rep_' if chamber_info == 'House of Representatives' else 'Sen_' if chamber_info == "Senate" else ''
            title_and_name = chamber + formatted_name
            title_and_name = title_and_name.replace('"','')
            url_list.append(title_and_name)
        result = sorted(url_list)
        return result
    except Exception as e:
        raise



if __name__ == "__main__":
    faktory_server_url = "tcp://localhost:7419"

    congress_client = CongressClient(api_key=CONGRESS_API_KEY)
    
    congress = congress_client.get_current_congress()

    title_names = congress_to_title_names(congress)

    names = sys.argv[1:] if len(sys.argv) > 1 else []
    for name in names:
        with Client(faktory_url=faktory_server_url, role="producer") as client:
            producer = Producer(client=client)
            job = Job(
                jobtype="search_legislator", args=(name,), queue="search_legislator"
            )
            producer.push(job)
            time.sleep(1)