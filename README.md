---
title : "Load Data"
weight : 51
---

Preparing and Loading the data to the Opensearch collection
---------------------------------------------------------------------

1. Navigate to **indexer** directory under /home/ec2-user/vector-engine-demos and run the python code that was stored in the directory. 
```bash
    cd  /home/ec2-user/vector-engine-demos/indexer
    python3 movies_loader_bedrock.py &
```

The python code we just ran will process and load the sample IMDB movies data to an OpenSearch Serverless index. While data is being loaded, let's navigate step by step in the code below and see what's happening behind the scenes.

Understanding the process:
----

The sample IMDB movie dataset has information pertaining to movies and details such as the title of the movie, plot, directors, actors, genre, etc. 

:::code{showCopyAction=false}
{"index": {"_index": "movies-index"}}
{"directors": ["Joseph Gordon-Levitt"], "release_date": "2013-01-18T00:00:00Z", "rating": 7.4, "genres": ["Comedy", "Drama"], "image_url": "http://ia.media-imdb.com/images/M/MV5BMTQxNTc3NDM2MF5BMl5BanBnXkFtZTcwNzQ5NTQ3OQ@@._V1_SX400_.jpg", "plot": "A New Jersey guy dedicated to his family, friends, and church, develops unrealistic expectations from watching porn and works to find happiness and intimacy with his potential true love.", "title": "Don Jon", "rank": 1, "running_time_secs": 5400, "actors": ["Joseph Gordon-Levitt", "Scarlett Johansson", "Julianne Moore"], "year": 2013}
{"index": {"_index": "movies-index"}}
{"directors": ["Ron Howard"], "release_date": "2013-09-02T00:00:00Z", "rating": 8.3, "genres": ["Action", "Biography", "Drama", "Sport"], "image_url": "http://ia.media-imdb.com/images/M/MV5BMTQyMDE0MTY0OV5BMl5BanBnXkFtZTcwMjI2OTI0OQ@@._V1_SX400_.jpg", "plot": "A re-creation of the merciless 1970s rivalry between Formula One rivals James Hunt and Niki Lauda.", "title": "Rush", "rank": 2, "running_time_secs": 7380, "actors": ["Daniel Br\u00c3\u00bchl", "Chris Hemsworth", "Olivia Wilde"], "year": 2013}
:::


Let's examine the python code to understand what's going on. This python file will be used to load data from your json file into the collection. The code uses Amazon Bedrock to compute text embeddings which can then be compared with cosine-similarity to find content with similar meaning. Amazon Bedrock is a fully managed service that offers a choice of high-performing foundation models (FMs) via a unified API. In this case, we're using the Amazon Titan Embeddings model to generate vector representations of our text data.

:::code{showCopyAction=false}
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import json
import boto3
import os
import time
import sys, getopt

# Set the vector size for Titan Embeddings model
vector_size = 1536  # Amazon Titan Embeddings model dimension

# Initialize Bedrock client
region = os.environ.get('AOSS_VECOTRSEARCH_REGION')
bedrock_runtime = boto3.client('bedrock-runtime', region_name=region)

def generate_embedding(text):
    """Generate embeddings using Amazon Bedrock Titan Embeddings model"""
    response = bedrock_runtime.invoke_model(
        modelId='amazon.titan-embed-text-v1',
        contentType='application/json',
        accept='application/json',
        body=json.dumps({
            'inputText': text
        })
    )
    
    response_body = json.loads(response['body'].read())
    return response_body['embedding']
:::


In the next section of the code, we are creating a new index and mapping the sample IMDB movies json file to knn_vector types. Short for its associated k-nearest neighbors algorithm, k-NN for Amazon OpenSearch Service lets you search for points in a vector space and find the "nearest neighbors" for those points by Euclidean distance or cosine similarity. Use cases include recommendations (for example, an "other songs you might like" feature in a music application), image recognition, and fraud detection.

:::code{showCopyAction=false}
# movies in JSON format
json_file_path = "sample-movies.json"

def full_load(index_name, client):
    
# if index_name exists in collection, don't run this again 
    # create a new index
    if not client.indices.exists(index=index_name):
        index_body = {
            "settings": {
                "index.knn": True
          },
          'mappings': {
            'properties': {
              "title": {"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},
              "v_title": { "type": "knn_vector", "dimension": vector_size },
              "plot": {"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},
              "v_plot": { "type": "knn_vector", "dimension": vector_size },
              "actors": {"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},
              "certificate": {"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},
              "directors": {"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},
              "genres": {"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},
              "genres": {"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},
              "image_url": {"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},
              "gross_earning": {"type":"float"},
              "metascore": {"type":"float"},
              "rating": {"type":"double"},
              "time_minute": {"type":"long"},
              "vote": {"type":"long"},
              "year": {"type":"long"}
            }
          }
        }

        client.indices.create(
          index=index_name, 
          body=index_body
        )
        time.sleep(5)
    
:::

In the next section of the code, we're actually loading the JSON data into the index. For this workshop, we'll only be encoding the title and plot using Amazon Bedrock's Titan Embeddings model. 

:::code{showCopyAction=false}
    actions = []
    i = 0
    j = 0
    action = {"index": {"_index": index_name}}

    # Read and index the JSON data
    with open(json_file_path, 'r') as file:
        for item in file:
            json_data = json.loads(item)
            if 'index' in json_data:
                continue

            # Generate embedding for title using Bedrock
            title = json_data['title']
            v_title = generate_embedding(title)
            json_data['v_title'] = v_title
    
            if 'plot' in json_data:
                # Generate embedding for plot using Bedrock
                plot = json_data['plot']
                v_plot = generate_embedding(plot)
                json_data['v_plot'] = v_plot
    
            # Prepare bulk request
            actions.append(action)
            actions.append(json_data.copy())
    
            if(i > 10 ):
                client.bulk(body=actions)
                print(f"bulk request sent with size: {i}")
                print(f"total docs sent so far: {j}")
                i = 0
                actions = []
            i += 1
            j += 1

:::

Finally, in this section you can see where we're passing the host, region and index to connect to the Opensearch collection and send the requests.

:::code{showCopyAction=false}
def main(argv):
    host = os.environ.get('AOSS_VECOTRSEARCH_ENDPOINT')
    region = os.environ.get('AOSS_VECOTRSEARCH_REGION')
    index = "opensearch_movies"
    service = 'aoss'

    credentials = boto3.Session().get_credentials()
    awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service,
                   session_token=credentials.token)

    # Build the OpenSearch client
    client = OpenSearch(
        hosts = [{'host': host, 'port': 443}],
        http_auth = awsauth,
        timeout = 300,
        use_ssl = True,
        verify_certs = True,
        connection_class = RequestsHttpConnection
    )
    print(f"OpenSearch Client - Sending to Amazon OpenSearch Serverless host {host} in Region {region} \n")
    full_load(index, client)

if __name__ == '__main__':
    main(sys.argv[1:])            
:::

3. To verify the document was succesfully indexed in the Amazon OpenSearch Serverless collection index, go to the collection and select **Monitor**. You'll be able to see all sorts of information such as indexing performance, storage, search performance, and errors. 

Congratulations! You now have an index full of movie data you can search. Please proceed to the next step to learn about some different types of queries available to you in OpenSearch
