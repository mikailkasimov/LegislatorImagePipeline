#Credit to https://www.kaggle.com/code/alaasweed/similarity-percentage-using-facenet

import numpy as np
import os
import json
import requests
from PreprocessingPipeline import PreprocessingPipeline
from FeatureExtractorPipeline import FeatureExtractionPipeline
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()
GOOGLE_CUSTOM_SEARCH_ENGINE_API_KEY = os.getenv("GOOGLE_CUSTOM_SEARCH_ENGINE_API_KEY")
GOOGLE_CUSTOM_SEARCH_ENGINE_ID = os.getenv("GOOGLE_CUSTOM_SEARCH_ENGINE_ID")
CONGRESS_API_KEY = os.getenv("CONGRESS_API_KEY")

def l2_normalize(vector):
    norm = np.linalg.norm(vector)
    return vector if norm == 0 else vector / norm

def calculate_cosine_similarity(feature1, feature2):
    """
    Computes the cosine similarity between two L2-normalized feature vectors.
    """
    f1 = l2_normalize(feature1)
    f2 = l2_normalize(feature2)
    cosine_sim = np.dot(f1, f2)
    return cosine_sim

def convert_cosine_similarity_to_percentage(cosine_sim):
    """
    Converts cosine similarity (which ranges from -1 to 1) into a percentage.
    For normalized face embeddings, values are typically between 0 and 1.
    """
    # Clamp to [0,1] and scale to percentage
    cosine_sim = np.clip(cosine_sim, 0, 1)
    return cosine_sim * 100



#------------------------------------------------------------------------------------------------#
"""
Parse json and get image links from Google Search Request

param:  dict
return: list of image links
"""
def get_links_from_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    image_links = [item["link"] for item in data.get("items", [])]
    return image_links


"""
Get image links from Google Search, limited to 100 links

param:  query: str
return: list of image links
"""
def get_image_links_from_google_search(query):
    all_links = []
    for start in range(1, 101, 10):
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": GOOGLE_CUSTOM_SEARCH_ENGINE_API_KEY,
            "cx": GOOGLE_CUSTOM_SEARCH_ENGINE_ID,
            "q": query,
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

""""
Downloads an image link to current directory or specified directory

param: image_links: list of str
param: output_dir: str
return: None
"""
def download_image_from_links(image_links, output_dir=""):
    # 1. Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving images to '{output_dir}/' directory.")
    print(f"Number of links to download: {len(image_links)}")

    # 2. Loop through all the image links
    for i, url in enumerate(image_links):
        try:
            # 3. Request the image content from the URL
            # The timeout prevents the script from hanging on a slow connection
            response = requests.get(url, timeout=15)

            # Raise an exception if the request returned an error (e.g., 404, 403)
            response.raise_for_status()

            # 4. Check if the content is actually an image
            content_type = response.headers.get('content-type')
            if not content_type or 'image' not in content_type:
                print(f"Skipping URL (not an image): {url}")
                continue

            # 5. Create a clean filename (e.g., image_001.jpg)
            # We'll try to get the extension from the content type (e.g., 'image/jpeg' -> 'jpeg')
            extension = content_type.split('/')[-1]
            if extension == 'jpeg': extension = 'jpg' # Standardize to jpg
            
            filename = f"image_{i+1:03d}.{extension}"
            filepath = os.path.join(output_dir, filename)

            # 6. Save the image to the local file
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            print(f"Successfully downloaded {filename}")

        except requests.exceptions.RequestException as e:
            # 7. Handle errors like timeouts, dead links, or other request issues
            print(f"Could not download {url}. Error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred for URL {url}: {e}")



"""
Searches google and downloads first 100 image results to output_dir.
If output_dir is not specified, saves to "output/"

param: Text query for google image search: str
param (Optional): output_dir: str
"""
def search_and_download(query, output_dir="output"):
    links = get_image_links_from_google_search(query)
    download_image_from_links(links, output_dir=output_dir)



"""
Gets all FaceNet embeddings from directory filled with images

param: image_directory, Directory with images
param: max_faces, Max number of faces allowed allowed. Skip the image if it
       has more faces than max_faces"
return: N x D list of FaceNet embeddings a singular face
"""
def get_embeddings_from_directory(image_directory, max_faces=None, target_size=(160,160)):
    embeddings = []
    paths = []
    preprocessing_pipeline = PreprocessingPipeline(target_size=target_size)
    feature_extraction_pipeline = FeatureExtractionPipeline(device='cuda')

    #For each file/image in the directory
    for filename in os.listdir(image_directory):
        image_path = os.path.join(image_directory, filename)
        try:
            #this executes if max_faces is not specified,
            if max_faces is not None:
                img = preprocessing_pipeline.load_image(image_path)
                number_of_faces = preprocessing_pipeline.detect_number_of_faces(img)
                if number_of_faces > max_faces:
                    print(f"Image '{image_path}' has {number_of_faces} faces.")
                    continue

            #this process can be optimized by calling preprocessing_pipeline.detect_and_align
            #preprocessing_pipeline.resize_image, and preprocessing_pipeline.normalize_image,
            #instead of calling load_image twice by calling preprocess.
            #i kept it like this for better readability
            preprocessed_img = preprocessing_pipeline.preprocess(image_path)
            features = feature_extraction_pipeline.extract_features(preprocessed_img)
            embeddings.append(features)
            paths.append(image_path)
        except Exception as e:
            print(f"Could not process {filename}. Error: {e}")            
    print(f"Processing complete.")
    print(f"Total number of embeddings extracted: {len(embeddings)}")
    return embeddings, paths


"""
Using the Congress API key, we get a full list of the senate
"""
def get_current_senate():
    all_members = []
    for i in range(0,750, 250):
        url = "https://api.congress.gov/v3/member?"
        args = {
            "api_key": CONGRESS_API_KEY,
            "currentMember": True,
            "limit": 250,
            "offset": i
        }
        response= requests.get(url, params=args)
        data = response.json()
        print(data)
        members = data["members"]
        for member in members:
            all_members.append(member)

    return all_members

"""
Compute cosine similarity of each vector to the centroid.
param: N x D list of vectors of dimension D

returns: 1D numpy array of similarities
"""
def cosine_sim_to_centroid(vectors):
    vectors = np.array(vectors)
    centroid = np.mean(np.array(vectors), axis=0).reshape(1, -1)  # make it 2D
    sims = cosine_similarity(vectors, centroid).ravel()
    return sims


