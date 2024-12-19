Anime Recommendation System

This project implements various machine learning methods for building an anime recommendation
system. The repository contains scripts and notebooks for collaborative filtering, content-based
recommendation, and baseline models. Follow the instructions below to set up the environment,
download the dataset, and run the code.

Directory Structure
    notebooks/: Contains Jupyter notebooks for implementing and testing different recommendation
    methods.
    src/: Contains Python scripts for data processing, evaluation, and models.
    dataset/: Handles data loading and preprocessing.
    eval/: Contains scripts for evaluation.
    models/: Contains implementations of recommendation models.

Prerequisites
    Install Python 3.
    Ensure you have Jupyter Notebook installed to run the notebooks.

Dataset
    Download the dataset from this Kaggle link.
    https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020/data
    Place the downloaded dataset into the src/dataset/ folder.

Running the Code
    Open a terminal and navigate to the project directory.
    Launch Jupyter Notebook:
        jupyter notebook
    Open and run the following notebooks sequentially based on the method you want to use:
        Data Analysis: notebooks/data_analysis.ipynb
        User-Previous-Based Baseline Testing: notebooks/test_user_previous_baseline.ipynb
        Genre-Based Baseline Testing: notebooks/test_genre_baseline.ipynb
        Collaborative Filtering: notebooks/collaborative_filtering.ipynb
        Content-Based Recommendation: notebooks/content_based.ipynb
    Each notebook contains the necessary code to load the data, train the models, and evaluate
    performance.

License
    This project is open-source and available under the MIT License.