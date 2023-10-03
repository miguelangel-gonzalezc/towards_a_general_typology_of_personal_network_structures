# Towards a General Typology of Personal Network Structures
## Miguel A. González-Casado, José Luis Molina and Angel Sánchez

### SUMMARY

This is the official repository that includes both the clean data and the Python class required to reproduce the results presented in the paper with the DOI: [ADD]

### HOW TO CITE THIS WORK:
[ADD]

### DESCRIPTION OF THE DATA FILES

The file 'dataset_with_outliers.csv' contains a dataset in which each row represents one of the 8,239 networks used for analysis, with each column containing a measured value for one of the 41 structural metrics.

The file 'id_map.csv' contains a dataset in which each row represents one of the original 9,115 networks. The 'new_id' column corresponds to the ID found in the index column of the 'dataset_with_outliers.csv' file, in order to map both files. This dataset provides information about the source dataset of the network, as well as details such as country, sex, or gender of the ego.

### TUTORIAL: HOW TO USE THE PYTHON CLASS 

#### STEP 1: SET UP

The first step is to set up the Python environment to prevent any potential issues related to packages and dependencies. This environment is presented in the 'STA_PythonEnvironment.yaml' file. You can configure it using the Anaconda interface or by adhering to the installation instructions for any other Python environment through the terminal.

The second step is to configure the folders following this diagram:

    Main Folder
         StructuralTypologyAnalyzer.py
         Data
             1.csv
             2.csv
              .
              .
              .
         Results

Basically, you need to create a folder containing the Python code 'StructuralTypologyAnalyzer.py', and two other folders: one called 'Data' and another called 'Results'. The Python class will look for your Personal Networks within the 'Data' folder, and will place the results of the analysis within the Results folder.

#### STEP 2: FORMAT OF THE INPUT DATA

There are two ways of providing your own corpus of Personal Networks to the Python class for analysis. 
##### OPTION 1

The first option is to provide the raw Adjacency Matrices for the **undirected** and **unweighted** Personal Networks in .csv format. Each Personal Network should have an individual file, distinguished by a unique number, and placed within the 'Data' folder. For example, if you have two Personal Networks for analysis, you would deposit the files '1.csv' and '2.csv,' each containing its respective Adjacency Matrix, into the 'Data' folder. Examples of these files have been included in this repository to aid readers with the file format.



