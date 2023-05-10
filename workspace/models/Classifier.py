import pdb
import os
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from common.config import *


class Classifier(object):
    def __init__(self, max_depth=None) -> None:
        self.dtree = tree.DecisionTreeClassifier(max_depth=max_depth)  # set max_depth for visualization
        
    def fit(self, X, y):
        self.dtree = self.dtree.fit(X=X, y=y)
        
    def predict(self, X):
        y = self.dtree.predict(X=X)
        ret = []
        for descrp_id in y:
            description = weather_descriptions[descrp_id]
            ret.append(description)
        return ret

    def predict_prob(self, X):
        y = self.dtree.predict_proba(X=X)
        raise NotImplementedError()
    
    def save_to_pkl(self, pkl_filepath):
        with open(pkl_filepath, 'wb') as f:
            pkl.dump(self.dtree, f)
    
    def load_from_pkl(self, pkl_filepath):
        with open(pkl_filepath, 'rb') as f:
            self.dtree = pkl.load(f)
    
    def visualize(self, figs_dir):
        dot_data = tree.export_graphviz(self.dtree, 
                                        out_file=os.path.join(figs_dir, "tree.dot"),
                                        feature_names=names_for_output_features,  
                                        class_names=list(weather_descriptions.values()),  
                                        filled=True, 
                                        rounded=True,  
                                        special_characters=True,)  

        # text_representation = tree.export_text(self.dtree,
        #                                        feature_names=names_for_output_features)
        # print(text_representation)
