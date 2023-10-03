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

The first option is to provide the raw Adjacency Matrices for the **undirected** and **unweighted** Personal Networks in .csv format. Each Personal Network should have an individual file, distinguished by a unique number, and placed within the 'Data' folder. For example, if you have two Personal Networks for analysis, you would deposit the files '1.csv' and '2.csv,' each containing its respective Adjacency Matrix, into the 'Data' folder. Examples of these files have been included in this repository within the PNs folder to aid readers with the file format.

##### OPTION 2

The second option is to provide directly a file like our 'dataset_with_outliers.csv', i. e., a file containing a dataset in which each row represents one of the networks used for analysis, with each column containing a measured value for one of the 41 structural metrics. This file should be placed within the 'Data' folder as well. This second option is the adequate one to reproduce our analyses using the 'dataset_with_outliers.csv' file. 

#### STEP 3: DEVELOP YOUR OWN STRUCTURAL TYPOLOGY OF PERSONAL NETWORK STRUCTURES

First of all, you need to compile the class 'StructuralTypologyAnalyzer.py'. Then, you can start constructing the typology using the following commands:

##### USAGE OF THE CLASS

This command loads the information of your Personal Networks. If 'dataset_path' is set to False, it will assume that the raw Adjacency Matrices of your Personal Networks are provided (Input Option 1). Thus, it will compute all the 41 structural metrics and construct a dataset similar to the 'dataset_with_outliers.csv' one. Conversely, if 'dataset_path' is set to True, this command will look for the file 'dataset_with_outliers.csv' directly (Input Option 2).

    STclass = StructuralTypology(dataset_path = False)

Next command will compute the outliers present in the data using three different algorithms, as explained in the main paper. 

    STclass.remove_outliers('medium')

The input variable accepts three values:
1) 'soft' - The code will consider as an outlier every network identified as one by at least one of the three algorithms. 
2) 'medium' - The code will consider as an outlier every network identified as one by at least two of the three algorithms. 
3) 'hard' - The code will consider as an outlier every network identified as one by the three algorithms.

Next command will produce a Pair-Plot similar to the one depicted in Section S2 of the Supplementary Material. Basically, we depict the coordinates of our networks in the subspace of Principal Components explaining 90% of the variance, identifying outliers with a different color, along with a loadings matrix that helps identify for which variables outliers present extreme values. Results are stored within the 'Results' folder. 

    STclass.assess_outliers_removal()

Next two commands help us do some Exploratory Data Analysis. They create the figures presented in Section S3 of the Supplementary Material (variables distributions and correlation matrix). 
    
    STclass.depict_variables_distribution()
    STclass.compute_and_plot_correlation_matrix()

Next command will check the two main assumptions necessary to perform Factor Analysis: KMO and Barlett's Sphericity Test.

    # We check the Factor Analysis assumptions
    STclass.check_factor_analysis_assumptions()

Next command will perform the non parametric Parallel Analysis to determine the number of factors/components to retain.

    STclass.perform_parallel_analysis()

Next commands perform Principal Component Analysis and Factor Analysis on the data, using the number of factors/components found by the PA. For the factor analysis we need to specify the rotation we want (0 - None | 1 - Varimax | 2 - Quartimax | 3- Equamax | 4 - Oblimin). 

    STclass.perform_principal_component_analysis()
    STclass.perform_factor_analysis(rot=1) 

Finally, these last commands perform the clustering analyses and save the results. 

    STclass.cluster_analysis()    
    STclass.plot_clustering_results()

The last command will take time and memory to execute, and will produce a large ammount of results in the 'Results' folder (it will give all the results for all numbers of clusters spanning 2-14). The names of the result files are designed to make it easy to classify these results in folders afterwards. 





