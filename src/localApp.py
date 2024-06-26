from random import sample
from PIL import Image
import pandas as pd
import zipfile
import os

import streamlit as st
from streamlit_extras.no_default_selectbox import selectbox
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities import SparseMatrixSimilarity

from models.utils.constants import *
from models.utils.mlutilities import gensim_tokenizer, get_recomendations_metadata


st.set_page_config(page_title="Papers Recomendation App")

model_name = "GrammarGuru"

def folder_exists(folder_path):
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        return True
    else:
        return False



def unzip_file(zip_file_path: str, modelname: str = model_name):
    if not folder_exists(f"models/{modelname}"):
        try:
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(f"models/")
            st.write("Model Zip file Extraction completed!.")
        except FileNotFoundError:
            raise("Error: The specified zip file was not found.")
        except zipfile.BadZipFile:
            raise("Error: The specified file is not a valid zip file.")


def generate_ramdom_examples(test_inputs, num_examples=4):
    # Select num_examples random choices from the test_inputs list
    selected_prompts = sample(test_inputs, num_examples)
    
    # Format the selected prompts into the desired string format
    examples = '### Examples of prompts\n'
    for prompt in selected_prompts:
        examples += f'- "{prompt}"\n'
    
    return examples


hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

image = Image.open('reports/figures/arxiv-logo.jpg')

st.sidebar.image(image , caption="Arxiv Papers Recomendation System",width = 256)
app_mode = st.sidebar.selectbox("Choose app mode", ["Generate Recomendations", "About this Project", "About Me"])

st.title("ResearchRadar")


@st.cache_data
def load_papers_corpus(path: str):
    return pd.read_parquet(path)

@st.cache_resource
def load_dict(path: str):
    dict_corpus = Dictionary.load(path)
    return dict_corpus

@st.cache_resource
def load_model(path: str ):
    tfidf_model = TfidfModel.load(path)
    return tfidf_model

@st.cache_resource
def load_sparse_matrix(path: str):
    similarities = SparseMatrixSimilarity.load(path)
    return similarities


if app_mode == "Generate Recomendations":    
    welcome_text = """    
    <div style="text-align: justify">Welcome to my paper recommendation project! This App is here to simplify your search for relevant scientific and academic papers. Our intelligent recommendation system, powered by  <strong>Machine Learning and natural language processing</strong>, analyzes keywords, abstracts, titles, authors, and more to provide personalized suggestions based on your interests. Say goodbye to information overload and let us guide you towards new horizons in your quest for knowledge. 
    """
    subjects = """
    Our model is trained to recommend papers in various domains, including:
    - Mathematics
    - Statistics
    - Electrical Engineering
    - Quantitative Biology
    - Economics

    Say goodbye to information overload and let us guide you towards **new horizons** in your quest for knowledge. Join us and discover a streamlined way to **explore, learn, and stay ahead** in your field. Welcome aboard!
    """
    st.markdown(welcome_text, unsafe_allow_html=True)
    st.markdown(subjects)
    st.divider()

    
    with st.container():
        examples = generate_ramdom_examples(TEST_INPUTS)
        st.markdown(examples)
        # st.divider()

    model_details = """
    ### Available Models
    - SemanticSherlock: trained on 100% of the data
    - LanguageLiberator: trained on 75% of the data
    - TextualTango: trained on 50% of the data
    - GrammarGuru: trained on 25% of the data
    """
    st.markdown(model_details)
    
    model_size = st.selectbox( 
        label='Select one Model Option:', 
        options=("GrammarGuru", "SemanticSherlock", "LanguageLiberator", "TextualTango"), 
        index=0
    )

    if model_size:
        with st.spinner('The model binaries are unziping ...'):
            zip_file_path = f"models/{model_size}.zip"
            unzip_file(zip_file_path)            

        with st.spinner('The model binaries are loading, please wait...'):

            df = load_papers_corpus(f"models/{model_size}/data/{model_size}.parquet.gzip") 
            dictionary = load_dict(f"models/{model_size}/dictionaries/{model_size}.dict") 
            model = load_model(f"models/{model_size}/tdidf/{model_size}.model") 
            matrix = load_sparse_matrix(f"models/{model_size}/similarities_matrix/{model_size}") 
            st.success('Models Loaded, yei!', icon="🚀")
        
        st.markdown("#### Generate Recommendations")
        # recs_number = st.slider("Enter the number of papers you need", min_value=1, max_value=10, value=3)
        query = st.text_input("Enter the description of the Paper you need (the more descriptive, the better)", value="")
        
        if query != "":
            cleaned_prompt = gensim_tokenizer(query)

            with st.spinner('Generating Recommendations ... '):
                results_df = get_recomendations_metadata(query=query, df=df, n=3, dictionary=dictionary, index=matrix, tfidf_model=model)
            
                ids = results_df['id'].to_list()
                titles = results_df['title'].to_list()
                authors = results_df['authors'].to_list()
                categories = results_df['categories'].to_list()
                abstracts = results_df['abstract'].to_list()
                release_date = results_df['update_date'].to_list()
                
                results = list(zip(ids, titles, authors, categories, abstracts, release_date))
                
                st.write("Your top 3 papers:")
                for result in results:
                    with st.container():
                        col1, col2 = st.columns([1,3])

                        with col1:
                            st.markdown(f"**Title:**")
                            st.markdown(f"**Author:**")
                            st.markdown(f"**Categories:**")
                            st.markdown(f"**release_date:**")
                            st.markdown(f"**Abstract:**")

                            
                        with col2:
                            st.write(f"Title: {result[1]}")
                            st.write(f"Author: {result[2]}")
                            st.write(f"Categories: {result[3]}")
                            st.write(f"release_date: {result[5]}")
                            st.write(f"Abstract: {result[4]}")
                            st.markdown(f"""[Paper Link](https://arxiv.org/abs/{result[0]})""")
                        st.divider()
                st.balloons()

        else:
            st.write("Please enter your prompt :)")
    
    else:
        st.warning('No option is selected')

    


    
elif app_mode == "About this Project":
    intro_text = """
    Welcome to my paper recommendation project! This application aims to simplify and speed up the process of finding relevant scientific and academic papers. It utilizes Machine Learning techniques and natural language processing to provide an effective solution for students, researchers, and general users.

    ### Key Features

    - **Intelligent Recommendation System:** The application uses advanced algorithms to analyze keywords, abstracts, titles, authors, and other metadata associated with each paper.
    - **Efficient Discovery Process:** By leveraging machine learning, the system identifies and suggests the most relevant papers based on the user's interests and areas of study.
    - **Comprehensive Analysis:** The recommendation system performs an exhaustive analysis of various aspects of each paper to ensure accurate and targeted recommendations.
    - **Time-saving Solution:** Instead of manually searching through vast amounts of information, users can rely on this application to streamline the paper discovery process.

    ### Available Models

    - SemanticSherlock: trained on 100% of the data
    - LanguageLiberator: trained on 75% of the data
    - TextualTango: trained on 50% of the data
    - GrammarGuru: trained on 25% of the data **(Deployed Version)**

    **Note:** Due to resource limitations on the free tier of Streamlit, only the GrammarGuru version of the model is available for deployment.


    ### Benefits

    - **Saves Time and Effort:** With the application's intelligent algorithms, users can avoid the challenges and time-consuming nature of searching for papers on their own.
    - **Increased Relevance:** By considering keywords, abstracts, titles, authors, and other metadata, the recommendation system provides users with highly relevant paper suggestions.
    - **Tailored to User Interests:** The system takes into account each user's interests and areas of study, ensuring that the recommended papers align with their specific needs.
    - **Accessible to All Users:** Whether you are a student, researcher, or general user, this application is designed to cater to a wide range of users' needs.

    ### Get Started

    Explore, discover, and reach new horizons in your search for knowledge with our paper recommendation application. Simplify your journey to finding relevant papers and stay ahead in your field.
    
    Take a look to this proyect in my [GitHub Repo](https://github.com/LewisPons/arxiv-paper-recommender-system)
    """



    st.markdown(intro_text)



elif app_mode == "About Me":
    st.title('About Me')
    mkdn = """
    <p style="text-align: justify;">Hey there! I'm  <strong>Luis Morales</strong>, a passionate data professional with a background in Actuarial Sciences and expertise in Data Engineering and Machine Learning. I love diving into complex data projects and helping organizations unlock the power of their data. From designing robust data pipelines to building powerful ML models, I enjoy the thrill of turning raw data into actionable insights. With my coding skills in Python and R, I'm always up for tackling challenging projects and learning new technologies along the way.
    Thank you for taking the time to learn a little bit about me!</p>
    """
    st.markdown(mkdn, unsafe_allow_html=True)
    st.success("Feel free to contact me here 👇 ")

    col1,col2,col3,col4 = st.columns((2,1,2,1))
    col1.markdown('* [LinkedIn](https://www.linkedin.com/in/luis-morales-ponce/)')
    col1.markdown('* [GitHub](https://github.com/LewisPons)')
    image2 = Image.open('reports/figures/profile.jpeg')
    st.image(image2, width=400)
    

