import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import graphviz
from common.common import *


class Classifier(object):
    def __init__(self) -> None:
        self.dtree = tree.DecisionTreeClassifier()  # set max_depth for visualization
        
    def fit(self, X, y):
        self.dtree = self.dtree.fit(X=X, y=y)
        
    def predict(self, X):
        return self.dtree.predict(X=X)

    def predict_prob(self, X):
        return self.dtree.predict_proba(X=X)
    
    def visualize(self):
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
