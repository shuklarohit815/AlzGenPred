# Written by Rohit Shukla (November, 2021)
# This script will classify the AD associated genes using network features.
# Please refer https://github.com/shuklarohit815/AlzGenPred for detailed tutorial.

import os
import argparse
from catboost import CatBoostClassifier
import pandas as pd
import pickle
import numpy as np

# Defining the variables for command line argument. One parameter (input file) is mandatory while others are optional.
if __name__ == "__main__":
    print(""" 
                    _    _      ____            ____               _
                   / \  | |____/ ___| ___ _ __ |  _ \ _ __ ___  __| |
                  / _ \ | |_  / |  _ / _ \ '_ \| |_) | '__/ _ \/ _` |
                 / ___ \| |/ /| |_| |  __/ | | |  __/| | |  __/ (_| |
                /_/   \_\_/___|\____|\___|_| |_|_|   |_|  \___|\__,_|
                 AlzGenPred: Alzheimer associated gene classification             
                  @Rohit Shukla and Tiratha Raj Singh, November, 2021                         
                     https://github.com/shuklarohit815/AlzGenPred    """)


    parser = argparse.ArgumentParser(description=" \
    This tool will classify the AD associated genes using the CatBoost algorithm. It will write the result\
    in the ""Output_AD_classification.csv"" file and will keep it in the same folder. Please provide the topogological features in the       \
    described format. If you are not aware about feature generation so please see the feature_generation_manual.docx \
     and ""topological_features.csv"" from https://github.com/shuklarohit815/AlzGenPred  \
        and prepare the input accordingly.")

    parser.add_argument("-f","--file", metavar="", required=True, help="Enter topological feature file in the described format") 
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-q", "--quiet", action = "store_true", help="print_quiet")
    group.add_argument("-v", "--verbose", action = "store_true", help="print_verbose")

    print("Examples:")
    print("")
    print("python AlzGenPred.py --file topological_features.csv")
    print("")
    print("python AlzGenPred.py -f topological_features.csv")
    print("")

args = parser.parse_args() # the args has all the command line variable

#Taking the value of given file name in a new string variable for furthur operation
feature_file = args.file

#The try keyword will stop the program to crash.
try:

#Reading the input files and will only select four features.
    test_data = pd.read_csv(feature_file)
    test_data_1 = test_data[["AverageShortestPathLength","ClosenessCentrality","NeighborhoodConnectivity","TopologicalCoefficient"]]

    #Threshold
    t = 0.57
#Function to write the result in CSV file.
    def display_result(cb_probs):
        ids = test_data["name"]
        ids=ids.tolist()
        output_file=open("Output_AD_classification.csv","w+")
        id_class=dict(zip(ids,cb_probs))
        col_headers=output_file.write("Gene name,Class,Probability,\n")
        for key in id_class:
            if id_class[key]>=t:
                line=str(key)+","+"AD"+","+str(id_class[key])+"\n"
                output_file.write(line)
            else:
                line=str(key)+","+"non-AD"+","+str(id_class[key])+"\n"
                output_file.write(line)

        output_file.close()

#Function to load the model and predict the result.    
    def predict_Alz():

        with open ("catboost_model.pkl", "rb") as f:
            model = pickle.load(f)
        cb_predictions = model.predict(test_data_1)
        cb_probs = model.predict_proba(test_data_1)[:, 1]
        display_result(cb_probs)
        return 
    predict_Alz()

#If the input file is not correct so it will print a massage.
except:
    print("The file format is not correct. Please double check the input file and match with the given topological features file.")


