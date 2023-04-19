import openai
import os
import numpy as np
import pandas as pd
import logging
import json
from training_data import data
from openai.embeddings_utils import get_embedding


# Fetch API key from env variable
openai.api_key = os.environ["OPENAI_API_KEY"]

# Load file path environment variable
# Note: path should be in the format directory_path + filename.csv
# Sample path : "../embeddings.csv"
embeddings_file_path = os.environ(PATH_TO_EMBEDDINGS)

# embedding model parameters
embedding_model = "text-embedding-ada-002"



def convert_to_dataframe():
    '''
    Returns training data as dataframe empty dataframe with columns pii and label
    '''
    columns = ["pii","label"]
    dataframe = pd.DataFrame(data,columns=columns)

    return dataframe

def generate_embeddings():
    '''
    This function feeds the training data PII types to open ai's embeddings endpoint
    and returns embeddings of length 8191 using the ada model
    '''

    # convert training data into a dataframe
    training_dataframe = convert_to_dataframe(data = data)

    logging.info("Generating embeddings for training data ...")
    try:
        training_dataframe["embeddings"]= get_embedding(training_dataframe["embeddings"],embedding_model)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return False

    # save embeddings in a csv so that they can be re-used
    logging.info("Saving file in current directory ...")
    training_dataframe.to_csv(embeddings_file_path+"/"+"pii_embeddings.csv")

    return True


def best_match(query_embedding):
    '''
    :param query_embedding:
    :return best batch between query embedding and pii embedding using cosine similarity :
    '''

    # reshape query embedding
    query_embedding = np.array(query_embedding).reshape(1 ,-1)

    try:

        # loop through embeddings to calculate similarity scores
        # store scores as tuples in the list in the form -> (pii, label,score)
        scores = [(embeddings_dataframe["pii"].values[i], embeddings_dataframe["label"].values[i],
                   openai.cosine_similarity(np.array(embeddings_dataframe["embedding"].values[i]).reshape(1, -1), query_embedding))
                  for
                  i, _ in enumerate(embeddings_dataframe["embedding"])]
        logging.info("Converted pii to embeddings successfully ")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return []

    # sort list
    sorted_scores = sorted(scores, key=lambda x: x[2], reverse=True)

    # return best match
    return sorted_scores[:1]


def chat_completion(input_query):
    '''
    :param input_query:
    :return Returns classification of PII as detected by GPT:
    '''
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": '''You are Personal Identifiable Information Detector. You have to determine the level of the PII.
    
                Level:
                - Medium
                - High
                - Secret // Used for security keys and number such as tax number, sin, credit card details and expiry.
            
                RESPONSE FORMAT
                ---
                ```json
                [{ "field": string // the field name
                "value": string // the field value
                "level": Level // the level should be one of (Medium, High, Secret)
                }]'''},
                        {"role": "user", "content": '''Return your response in the RESPONSE FORMAT:
            
                {input_query}: 
            
                  '''.format(input_query=input_query)}
                    ]
                )
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return False


    # parse response
    response = completion["choices"][0]["message"]["content"]

    # Remove the backticks
    response = response.replace('```json\n', '').replace('\n```', '')
    predicted_class = json.loads(response)[0]["level"]

    return predicted_class

def is_stored_pii(query):
    '''
    :param query:
    :return True or False depending on whether query is found in training data:
    '''

    try:
        # Check if query is present in labelled data
        mask = embeddings_dataframe['PII'] == query

        # Retrieve the age for the rows where 'USA' is present in the 'Country' column
        data_row = embeddings_dataframe.loc[mask, 'Label']

        # Extract just the number 25 from the Series
        classification = data_row.iloc[0]
    except IndexError:
        logging.info("Query not found in training data")
        return False

    return classification




def classify(query,embeddings_dataframe):
    '''
    :param query:
    :return: classification of pii level - high, medium, secret
    '''



    # convert input to embedding
    query_embedding = get_embedding(query, embedding_model)

    if


    # Assumption - query is not present





try:
    # load embeddings
    embeddings_dataframe = pd.read_csv(embeddings_file_path)
    embeddings_dataframe["embedding"] = embeddings_dataframe.embedding.apply(lambda x: json.loads(x))
except FileNotFoundError:
    logging.warning("Embeddings file not found. Attempting to generate embeddings ...")
    embeddings_dataframe = generate_embeddings()



# take in a query via command line
query = input("Enter a query: ")



predict = openai.Classification.create(
    search_model="davinci",
    model="davinci",
    examples = training,
    query = query,
    labels = ["High", "Medium", "Secret", "None"],
).label.lower()

print("The classification is: " + predict)