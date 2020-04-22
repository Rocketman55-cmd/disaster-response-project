### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Description](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

The code should run using Python versions 3.*. Necessary libraries include pandas, numpy, plotly, flask, scikit-learn, nltk, and sqlalchemy.

## Project Motivation<a name="motivation"></a>

This is a Udacity Nanodegree project. The project was to create a etl pipeline and a trained scikit model to understand and categorize disaster messages on a web app with visualizations. 

## File Descriptions <a name="files"></a>

Many files are used for this project. 

- README.md - this file that describes the project 
- run.py - python file that starts the web app with visualizations 
- go.html - along with master.html, this is for the web app visualizations
- master.html - along with go.html, this is for the web app visualizations
- ETL Pipeline Preparation.ipynb - jupyter notebook that is a precursor to process_data.py and cleans the data
- disaster_categories.csv - data file with the categories of disaster messages 
- disaster_messages.csv - data file that contains messages from disasters
- process_data.py - file that cleans the data and stores it in a sql database file to prepare for the train_classifier.py
- ML Pipeline Preparation.ipynb - jupyter notebook that is a precursor to the train_classifier.py and creates the pipeline 
- train_classifier.py - python file that creates the trained pipeline and stores it as a pickle file


## Results<a name="results"></a>

The code and documentation can be found at this available GitHub [post](https://github.com/Rocketman55-cmd/disaster-response-project).

An etl pipeline is created with a trained model and it is displayed on a web app. The process_data.py cleans the data and stores it in a sql database file. Then the sql database file is loaded into the train_classifier.py and a pipeline is created to categorize the disaster messages.
The pipeline created then gets stored into a pickle file which then is used for the run.py. The run.py file displays three visualizations and the webpage can be used to classify messages. 

python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db

python train_classifier.py ../data/DisasterResponse.db classifier.pkl

python run.py

env|grep WORK

The above commands will be necessary to run the project. First, the process_data.py needs to run and the output is a sql database file. Then, the train_classifier.py will run using the sql database file.
A trained model is created and is stored in a pickle file. Finally, the run.py will run using the pickle file. In one terminal, the run.py should be ran and in another terminal, the command env|grep WORK should be ran.
In the terminal running the env|grep WORK, use the workspace domain and workspace ID to access the webpage. It should be in the form of: https://(workspaceID)-3001.(workspacedomain).
The webpage should appear with the visualizations.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

I am the author of this project and acknowledgements go to Udacity for this project. 