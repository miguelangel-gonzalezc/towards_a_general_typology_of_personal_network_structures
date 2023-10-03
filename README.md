# towards_a_general_typology_of_personal_network_structures

This is the official repository containing both the clean data and the Python Class necessary to reproduce the results from the paper with DOI: [add]

The file 'dataset_with_outliers.csv' contains a dataset in which each row represents one of the 8,239 networks used for analysis, and each column contains the value measured for each one of the 41 structural metrics.

The file 'id_map.csv' contains a dataset in which each row represents one of the original 9,115 networks. The column 'new_id' matches with the ID in the index column from the file 'dataset_with_outliers.csv'. This dataset contains information about the specific dataset from which the network was extracted, and the country, sex or gender of the ego. 

TUTORIAL: HOW TO USE THE PYTHON CLASS StructuralTypologyAnalyzer.py AND DEVELOP A STRUCTURAL TYPOLOGY FOR ANY DATASET OF PERSONAL NETWORKS

  1) The first step is to set up the Python environment, to avoid problems with packages and dependencies.
