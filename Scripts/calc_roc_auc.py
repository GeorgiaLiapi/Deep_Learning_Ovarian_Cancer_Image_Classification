# -*- coding: utf-8 -*-
"""
     **DISCLAIMER of Copyrights:
       The following function is part of the original plot_roc_curve() from the 'plotters.py' (lines 186-330) that belongs to: 
       reiinakano/scikit-plot repository (https://github.com/reiinakano/scikit-plot) **
     
     Usage in the current study: 
              User: Georgia Liapi, Master Student in Systems Biology, Maastricht University
     Academic Year: 2019-2020
           Project: 'Deep learning-based prediction of molecular subtypes in human ovarian cancer'
           Utility: This funtion (renamed) generates the true positive rate, the false positive rate and the ROC AUC per class (ovarian cancer subtype, multi-class classification)
                    for the data used for predictions in the Train_Val_Pred.py or the Train_whole_Test.py
                    This funtion is called in the Train_Val_Pred.py or the Train_whole_Test.py scripts
                    
             Note:  The following lines 26-46 come from the lines 243 to 260 of the original plot_roc_curve() of the reiinakano/scikit-plot repository. 
"""

# Import the required libraries
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


def roc_auc(y_true, y_probas, Idx, classes_to_plot=None):
              
              y_true = np.array(y_true)
              y_probas = np.array(y_probas)
          
              classes = np.unique(y_true)
              probas = y_probas
          
              if classes_to_plot is None:
                  classes_to_plot = classes
          
              fpr_dict = dict()
              tpr_dict = dict()
              roc_auc  = dict()
              
              indices_to_plot = np.in1d(classes, classes_to_plot)
              
              for i, to_plot in enumerate(indices_to_plot):
                   
                  fpr_dict[i], tpr_dict[i], _ = roc_curve(y_true, probas[:, i], pos_label=classes[i])
                  
                  if to_plot:
                      roc_auc[i] = auc(fpr_dict[i], tpr_dict[i])
                      
              return fpr_dict, tpr_dict, roc_auc;