# -*- coding: utf-8 -*-
"""
     **DISCLAIMER of Copyrights:
       The following function is the original plot_roc_curve() from the 'plotters.py' (lines 186-330) that belongs to: 
       reiinakano/scikit-plot repository (https://github.com/reiinakano/scikit-plot) **
     
     Usage in the current study: 
              User: Georgia Liapi, Master Student in Systems Biology, Maastricht University
     Academic Year: 2019-2020
           Project: 'Deep learning-based prediction of molecular subtypes in human ovarian cancer'
           Utility: This funtion generates a ROC curve per class (ovarian cancer subtype, multi-class classification for the data used for predictions
                    in the Train_Val_Pred.py or the Train_whole_Test.py
                    This funtion is called inside Train_Val_Pred.py or the Train_whole_Test.py scripts
                    
              Note: The adjusted plot_roc() is utilized for the current project, with some added arguments (Data, Idx) and minor changes in the current lines 61-62.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.preprocessing import label_binarize
import pandas as pd
from collections import Counter


def plot_roc(y_true, y_probas, Data, Idx, title='ROC Curves',   plot_micro=False, plot_macro=False, classes_to_plot=None,
                   ax=None, figsize=(14,7), cmap='tab20c', title_fontsize="large", text_fontsize="large"):

              y_true = np.array(y_true)
              y_probas = np.array(y_probas)
          
              classes = np.unique(y_true)
              probas = y_probas
          
              if classes_to_plot is None:
                  classes_to_plot = classes
          
              if ax is None:
                   
                  # ------ Plot the training and validation metrics ---------
                  plt.style.use('seaborn-colorblind') 
                  fig, ax = plt.subplots(1, 1, figsize=figsize)
          
              ax.set_title(title, fontsize=title_fontsize)
          
              fpr_dict = dict()
              tpr_dict = dict()
          
              indices_to_plot = np.in1d(classes, classes_to_plot)
              
              for i, to_plot in enumerate(indices_to_plot):
                   
                  fpr_dict[i], tpr_dict[i], _ = roc_curve(y_true, probas[:, i], pos_label=classes[i])
                  
                  if to_plot:
                      roc_auc = auc(fpr_dict[i], tpr_dict[i])
                   
                      color = plt.cm.get_cmap(cmap)(float(i) / len(classes))
                      ax.plot(fpr_dict[i], tpr_dict[i], lw=3, color=color,
                              label='ROC curve of class {0} (area = {1:0.2f})'
                                    ''.format(sorted(Counter(Data['Class']))[i], roc_auc))
          
              if plot_micro:
                  binarized_y_true = label_binarize(y_true, classes=classes)
                  if len(classes) == 2:
                      binarized_y_true = np.hstack(
                          (1 - binarized_y_true, binarized_y_true))
                  fpr, tpr, _ = roc_curve(binarized_y_true.ravel(), probas.ravel())
                  roc_auc = auc(fpr, tpr)
                  ax.plot(fpr, tpr,
                          label='micro-average ROC curve '
                                '(area = {0:0.2f})'.format(roc_auc),
                          color='teal', linestyle='--', linewidth=3.5)
          
              if plot_macro:
                  # Compute macro-average ROC curve and ROC area
                  # First aggregate all false positive rates
                  all_fpr = np.unique(np.concatenate([fpr_dict[x] for x in range(len(classes))]))
          
                  # Then interpolate all ROC curves at this points
                  mean_tpr = np.zeros_like(all_fpr)
                  for i in range(len(classes)):
                      mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
          
                  # Finally average it and compute AUC
                  mean_tpr /= len(classes)
                  roc_auc = auc(all_fpr, mean_tpr)
          
                  ax.plot(all_fpr, mean_tpr,
                          label='macro-average ROC curve '
                                '(area = {0:0.2f})'.format(roc_auc),
                          color='darkblue', linestyle='-.', linewidth=3.5)
          
              ax.plot([0, 1], [0, 1], 'k--', lw=2)
              ax.set_xlim([0.0, 1.0])
              ax.set_ylim([0.0, 1.05])
              ax.set_xlabel('False Positive Rate', fontsize=text_fontsize)
              ax.set_ylabel('True Positive Rate', fontsize=text_fontsize)
              ax.tick_params(labelsize=text_fontsize)
              ax.legend(loc='lower right', fontsize=text_fontsize)
              
              return ax;