# Disaster Response Pipeline Project

This Project is part of Data Science Nanodegree Program by Udacity in collaboration with Figure Eight. The initial dataset contains pre-labelled tweet and messages from real-life disaster. The aim of the project is to build a Natural Language Processing tool that categorize messages.

The Project is divided in the following Sections:

    1. Data Processing, ETL Pipeline to extract data from source, clean data and save them in a proper databse structure
    2. Machine Learning Pipeline to train a model able to classify text message in categories
    3. Web App to show model results in real time.


  ### Code and data

-  process_data.py: This code extracts data from both CSV files: messages.csv (containing message data) and categories.csv (classes of messages) and creates an SQLite database containing a merged and cleaned version of this data.
-  train_classifier.py: This code takes the SQLite database produced by process_data.py as an input and uses the data contained within it to train and tune a ML model for categorizing messages. The output is a pickle file containing the fitted model. Test evaluation metrics are also printed as part of the training process.
-  ETL Pipeline Preparation.ipynb: The code and analysis contained in this Jupyter notebook was used in the development of process_data.py. process_data.py  automates this notebook.
-  ML Pipeline Preparation.ipynb: The code and analysis contained in this Jupyter notebook was used in the development of train_classifier.py. In particular, it contains the analysis used to tune the ML model and determine which model to use. train_classifier.py automates the model fitting process contained in this notebook.
-  disaster_messages.csv, disaster_categories.csv contain sample messages (real messages that were sent during disaster events) and categories datasets in csv format.
-  templates folder: This folder contains all of the files necessary to run and render the web app.
-  custom_transformer.py contains custom functions that were used in ML Pipeline Preparation.ipynb so to find best way of model tuning.


### Instructions:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Screenshots

![result1](https://github.com/avkolte/Disaster-Response-Pipeline-Project/blob/master/results/result1.png)
![result2](https://github.com/avkolte/Disaster-Response-Pipeline-Project/blob/master/results/result2.png)
