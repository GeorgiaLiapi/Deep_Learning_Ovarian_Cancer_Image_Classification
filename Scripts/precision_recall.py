"""
     **DISCLAIMER of Copyrights:
       The following function is the original plot_precision_recall_curve() from the 'plotters.py' (lines 422-541) that belongs to: 
       reiinakano/scikit-plot repository (https://github.com/reiinakano/scikit-plot) **
     
     Usage in the current study: 
              User: Georgia Liapi, Master Student in Systems Biology, Maastricht University
     Academic Year: 2019-2020
           Project: 'Deep learning-based prediction of molecular subtypes in human ovarian cancer'
           Purpose: This funtion generates a Precision-Recall curve per class (ovarian cancer subtype, multi-class classification) for the data used for predictions
                    in the Train_Val_Pred.py or the Train_whole_Test.py
                    This funtion is called in the Train_Val_Pred.py or the Train_whole_Test.py scripts
                    
                    Note: The adjusted plot_roc() is utilized for the current project, with some added arguments (Data, Idx) and minor changes in the current lines 74-75.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
import pandas as pd
from collections import Counter


def plot_precision_recall_curve(y_true, y_probas, Data, Idx, title='Precision-Recall Curve', curves=('micro', 'each_class'), ax=None, figsize=(14,7), cmap='tab20c',
                                title_fontsize="large", text_fontsize="large"):
    
    y_true = np.array(y_true)
    y_probas = np.array(y_probas)

    classes = np.unique(y_true)
    probas = y_probas

    if 'micro' not in curves and 'each_class' not in curves:
        raise ValueError('Invalid argument for curves as it '
                         'only takes "micro" or "each_class"')

    # Compute Precision-Recall curve and area for each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(len(classes)):
        precision[i], recall[i], _ = precision_recall_curve(
            y_true, probas[:, i], pos_label=classes[i])

    y_true = label_binarize(y_true, classes=classes)
    if len(classes) == 2:
        y_true = np.hstack((1 - y_true, y_true))

    for i in range(len(classes)):
        average_precision[i] = average_precision_score(y_true[:, i],
                                                       probas[:, i])

    # Compute micro-average ROC curve and ROC area
    micro_key = 'micro'
    i = 0
    while micro_key in precision:
        i += 1
        micro_key += str(i)

    precision[micro_key], recall[micro_key], _ = precision_recall_curve(y_true.ravel(), probas.ravel())
    average_precision[micro_key] = average_precision_score(y_true, probas, average='micro')

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_title(title, fontsize=title_fontsize)

    if 'each_class' in curves:
        for i in range(len(classes)):
            color = plt.cm.get_cmap(cmap)(float(i) / len(classes))
            ax.plot(recall[i], precision[i], lw=2,
                    label='Precision-recall curve of class {0} '
                          '(area = {1:0.3f})'.format(sorted(Counter(Data['Class']))[i],
                                                     average_precision[i]), color=color)

    if 'micro' in curves:
        ax.plot(recall[micro_key], precision[micro_key],
                label='micro-average Precision-recall curve '
                      '(area = {0:0.3f})'.format(average_precision[micro_key]),
                color='navy', linestyle=':', linewidth=4)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.tick_params(labelsize=text_fontsize)
    ax.legend(loc='best', fontsize=text_fontsize)

    return ax
