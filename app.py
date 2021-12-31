from numpy.core.arrayprint import _make_options_dict
import streamlit as st
import pickle
import pandas as pd
import scipy.spatial
import numpy as np
import os, json
import glob
import re
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from tokenizers import Tokenizer
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)
import pandas as pd
import torch
import random
import itertools
import pickle
import time

start = time.time()

st.set_page_config(layout="wide", page_title="Semantic Search for defining Indicators!", page_icon="🐞")

st.header("🐞 Semantic Search for defining Indicators!")
st.subheader('AI-Growth-Lab AAU')

# embedder = SentenceTransformer('bert-base-nli-stsb-mean-tokens')

@st.cache
def convert_df_to_csv(df):
  # IMPORTANT: Cache the conversion to prevent computation on every rerun
  return df.to_csv().encode('utf-8')

def get_top_n_similar_patents_df(new_claim, claim_embeddings):
    search_hits_list = []
    search_hits = util.semantic_search(new_claim, claim_embeddings, 10000, 5000000, 20)
    # save similar patents info
    top_doc_id = []
    top_similarity_scores = []
    for item in range(len(search_hits[0])):
        top_doc_id.append(search_hits[0][item]['corpus_id'])
        top_similarity_scores.append(search_hits[0][1]['score'])
        
    top_n_similar_patents_df = pd.DataFrame({
        'top_doc_id': top_doc_id,
        'cosine_similarity': top_similarity_scores
    })
    return top_n_similar_patents_df

st.write("A Sample of the CSV")
df_sample = pd.read_csv('paragraphs_text.csv')
st.dataframe(df_sample.head(2))

uploaded_file_docs = st.file_uploader("Upload CSV file of Docs!")
if uploaded_file_docs is not None:    
    #read csv
    df_docs = pd.read_csv(uploaded_file_docs)
    df_docs.head()

st.markdown('<h1 style="background-color: gainsboro; padding-left: 10px; padding-bottom: 20px;">Indicator Search Engine</h1>', unsafe_allow_html=True)
df_example = st.text_input('', help='Enter the search string and hit Enter/Return')

# uploaded_file_example = st.file_uploader("Upload CSV file of Indicator example!")
# if uploaded_file_example is not None:    
#     #read csv
#     df_example = pd.read_csv(uploaded_file_example)
#     df_example.head()

if st.button("Search"):

    model = SentenceTransformer('all-MiniLM-L6-v2')

    sentences_example = df_example
    sentences_docs = df_docs.paragraph_text.to_list()

    embeddings_example = model.encode(sentences_example)
    embeddings_docs = model.encode(sentences_docs)

    df_results = get_top_n_similar_patents_df(embeddings_example, embeddings_docs)
    st.table(df_results)
    # st.download_button("Press to Download", df_results,"file.csv","text/csv",key='download-csv')
    st.download_button(label='📥 Download Current Result', data=convert_df_to_csv(df_results), file_name= 'df_results.csv')
    
    #Store sentences & embeddings on disc
    with open('embeddings_docs.pkl', "wb") as fOut:
        pickle.dump({'sentences': sentences_docs, 'embeddings': embeddings_docs}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

    #Store sentences & embeddings on disc
    with open('embeddings_example.pkl', "wb") as fOut:
        pickle.dump({'sentences': sentences_example, 'embeddings': embeddings_example}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

    #Load sentences & embeddings from disc
    with open('embeddings_example.pkl', "rb") as fIn:
        stored_data = pickle.load(fIn)
        stored_sentences = stored_data['sentences']
        stored_embeddings = stored_data['embeddings']