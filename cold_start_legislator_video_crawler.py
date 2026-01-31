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

if __name__ == "__main__":
    faktory_server_url = "tcp://localhost:7419"

    name='Rep_Ilhan_Omar'
    query='Rep. Ilhan Omar interview'
    max_results=100

    with Client(faktory_url=faktory_server_url, role="producer") as client:
        producer = Producer(client=client)
        job = Job(
            jobtype="search_legislator", args=(name,query,max_results), queue="search_legislator"
        )
        producer.push(job)
