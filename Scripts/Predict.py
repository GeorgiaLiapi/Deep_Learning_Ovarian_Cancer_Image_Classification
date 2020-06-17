# -*- coding: utf-8 -*-
"""
        Author: Georgia Liapi, Master Student in Systems Biology, Maastricht University
 Academic Year: 2019-2020
       Purpose: This is a class used by the Train_Val_Pred.py or the Train_whole_Test.py script, and
                generates results after predicting on the given data fold or the testing data of the current dataset
                
                NOTE : The type and number of the results are given in the README file
"""
# Import the required libraries
import numpy as np
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from collections import Counter


class Predict(object):
     
     # DataType is either 'last_fold' or 'TestData'
     def __init__(self, Data, model, network, DataType, Generator, Idx , Results_Dirs, cvtype, experiment, Project_Dir):
              
          self.Data                   = Data
          self.DataType               = DataType
          self.Generator              = Generator
          self.model                  = model
          self.network                = network 
          self.Project_Dir            = Project_Dir
          self.Idx                    = Idx
          self.experiment             = experiment
          self.cvtype                 = cvtype
          self.Results_Dirs           = Results_Dirs
          
     def predictions(self):
          
          # *************************** Predicting  ***************************

          # Define the steps per epoch for the predict generator
          steps  =  self.Generator.n
          
          # Explain the array of the classes                  
          Labels = ( self.Generator.class_indices)
          class_labels = list( self.Generator.class_indices.keys())
          
          # ************************  Predict *********************************
          self.Generator.reset()
          Predictions = self.model.predict_generator( self.Generator,verbose=1, steps = steps, workers=4, max_queue_size=20)
          
          # Generate the hard predictions from the soft predictions (gives the position of the class with the highest probability)
          Pred_Class_Indcs=np.argmax(Predictions,axis=1)
          
          #--------------------------------------------------------------------
          #                     IMAGE  TILES' LEVEL
          #--------------------------------------------------------------------
          # Extract the report from classification
          report = classification_report(self.Generator.classes, Pred_Class_Indcs, target_names=class_labels, output_dict=True)
            
          # Save the classification report
          Report= pd.pandas.DataFrame(report).astype(object).transpose()         
          Project_Dir1= self.Results_Dirs + '\\' + self.experiment + '_' + self.network + '_Report_' + self.DataType + self.cvtype + str(self.Idx) + '.xlsx'
          Report.to_excel(Project_Dir1, index= True)              
          
          # Generate the confusion matrix--------------------------------------         
          cm=confusion_matrix( self.Generator.classes,Pred_Class_Indcs)
                    
          fig1= plt.figure(figsize=(5*3.13,3*3.13))
          Ax = fig1.add_subplot(111)       
          Cax = Ax.matshow(cm)
          
          plt.title('Ovarian Cancer_{}_Prediction results on the level of tiles\n({}{}{})'.format(self.experiment,self.network,self.cvtype,self.Idx,), fontsize=8, y=1.109)              
          fig1.colorbar(Cax, orientation='vertical')
          Ax.set_xticklabels([''] + list(self.Generator.class_indices.keys()), rotation=33, fontsize=9, y=1.001)
          Ax.set_yticklabels([''] + list(self.Generator.class_indices.keys()), fontsize=9)

          plt.ylabel('True Label')
          plt.xlabel('Predicted Label')
              
          fig1.tight_layout()
          fig1= plt.gcf()
          plt.draw()
          fig1.savefig(self.Results_Dirs + '\\' + self.experiment + '_'+ self.network + '_CM_Tiles_' + self.DataType + self.cvtype + str(self.Idx) + ".png", dpi=900)      
            
          # Display the confusion matrix on the level of the tiles
          cm_df = pd.DataFrame(confusion_matrix( self.Generator.classes,Pred_Class_Indcs), columns=class_labels, index=class_labels)
          cm_df.index.name = 'True Labels'
          Project_Dir2= self.Results_Dirs + '\\' + self.experiment +'_'+ self.network + 'CM_Tiles_' + self.DataType + self.cvtype + str(self.Idx) + '.xlsx'
          cm_df.to_excel(Project_Dir2, index= True)
          
          # Save the soft and hard predictions
          Predicted= pd.DataFrame(data={"Predicted": list(Predictions), 'Position': Pred_Class_Indcs})
          Predicted['Predicted_Label'] = pd.Series()               
          
          # Add to the Predicted table the predicted label name and the True Labe----------------------------------------------------------------
          for pos, el in enumerate(Predicted['Position']):
               
               Predicted['Predicted_Label'].loc[pos] = class_labels[el]
               
          Predicted['True_Labels'] = self.Data['Class'].values
                          
          # Filter for the True Positives and create a 5th column on the 'Predicted' DF ---------------------------------------------------------
          Predicted['True_Positives'] = pd.Series()
          
          for Pos,El in enumerate(Predicted.iterrows()):                     
               # If Predicted label is the same as the True
               if El[1][2] == El[1][3]:
                    
                    Predicted['True_Positives'].loc[Pos] = El[1][0].max()
               else:
                    
                    Predicted['True_Positives'].loc[Pos] = 'False'
          
          # Set the indices to those of the primary table                         
          Predicted.set_index(pd.Series(self.Data.index), inplace=True)
          
          # Save the dataframe with the filtered predictions               
          Project_Dir0= self.Results_Dirs + '\\' + self.experiment + '_'+ self.network + '_Predicted_' + self.DataType + self.cvtype + str(self.Idx) + '.xlsx'
          Predicted.to_excel(Project_Dir0)
          
          #--------------------------------------------------------------------
          #                      PATIENT LEVEL
          #--------------------------------------------------------------------
          
          # Filter out only the true positive predictions per patient
          Predicted_Filtered = Predicted             
          Predicted_Filtered = Predicted_Filtered[~Predicted_Filtered['True_Positives'].isin(['False'])]
          Predicted_Filtered['True_Positives'] =  Predicted_Filtered['True_Positives'].astype(float)
          
          # Instantiate a table containing the correctly and falsely predicted labels, 
          # as well as the classification accuracy 'per patient'
          Classification_Metrics = pd.DataFrame(columns = ['Correct_Predictions', 'False_Predictions', 'Classification_Accuracy', 'Mean_Probability', 'True_Label'], index=list(Predicted.index.unique()))
          
          # Fill in the columns with the corresponding metrics per patients' results
          for indx in Predicted.index.unique():
               
               if indx in Predicted_Filtered.index.unique():
                    
                    Classification_Metrics['Correct_Predictions'][indx]     =  len(Predicted.loc[indx]) - Predicted['True_Positives'][indx].value_counts()[0]
                    Classification_Metrics['False_Predictions'][indx]       =  Predicted['True_Positives'][indx].value_counts()[0]
                    Classification_Metrics['Classification_Accuracy'][indx] =  Classification_Metrics['Correct_Predictions'][indx]/ len(Predicted.loc[indx])                                                                                                                           
                    Classification_Metrics['True_Label'][indx]              =  Predicted['True_Labels'][indx].values[0]
                    
                    """
                    For each patient, I average the probabilities that correspond to her positives (correct predicitons) and put this in the 'Mean_Probability' column in the patient's row
                    To calculate the mean probability out of all the predicted labels assigned to a patient, I leave out the wrong predictions (which I had set as 'False',
                    in the unfiltered 'Predicted' datarame, creating a filtered table, 'Predicted_Filtered')
                    """
                    Prob = Predicted_Filtered['True_Positives'][indx]
                    Classification_Metrics['Mean_Probability'][indx]  =  np.mean(Prob)
          
          # Save the table in the current dataset results folder
          Classification_Metrics.to_excel(self.Results_Dirs + '\\' + self.experiment + '_'+ self.network + '_Classification_Metrics_' + self.DataType + self.cvtype + str(self.Idx) + '.xlsx')
          
          #Count the number of predicted classes per patient
          NumOfClassPerID = pd.DataFrame(data={})
          
          # The NumOfClassPerID is a table, where each row refers to a patient id (index), and each column is a class in the current dataset
          # where the number of the predicted tiles per class for each patient's images tiles is given (no control for false predictions, yet) 
          for el, els in enumerate(Predicted['Position'].index.unique()):
               
               NumOfClassPerID = NumOfClassPerID.append(Counter(Predicted['Position'][els]), ignore_index=True)    
               
          NumOfClassPerID = NumOfClassPerID[list(Labels.values())]
          NumOfClassPerID.columns = list(Labels.keys())
          
          NumOfClassPerID.fillna(0, inplace=True)
          NumOfClassPerID = NumOfClassPerID.set_index(Predicted['Position'].index.unique())
          NumOfClassPerID.fillna(0, inplace=True)
          
          NumOfClassPerID['True_Label'] = pd.Series()
          
          # Adding a column carrying the true label for each patient (row)
          for indx in Predicted.index.unique():
               
               NumOfClassPerID['True_Label'][indx] =  Predicted['True_Labels'][indx].values[0]
               
          Project_Dir1= self.Results_Dirs + '\\' + self.experiment + '_'+ self.network + '_NumOfClassPerID_' + self.DataType + self.cvtype + str(self.Idx) + '.xlsx'
          NumOfClassPerID.to_excel(Project_Dir1, index= True)
                        
          # Instatiate a Confusion Matrix on the patient level------------------------------                    
          CM_Patients = pd.DataFrame(data={}, index= list(sorted(NumOfClassPerID.columns[:-1])), columns = list(sorted(NumOfClassPerID.columns[:-1])))            
          CM_Patients.fillna(0, inplace=True) 
          
          # -------------------------------------------------------------------
          # Creating the confusion matrix on the patient level
          # -------------------------------------------------------------------
          
          # ROWS: TRUE LABELS
          # COLUMNS: PREDICTED LABELS
             
          # FILLING THE CONFUSION MATRIX ON THE PATIENT LEVEL
          for x in NumOfClassPerID.iterrows():
               
               # Searching for the class with the highest aggregated number of predictions in each row (patient) of the NumOfClassPerID
               # (compared to the number of predictions in the other class, for the current patient). 
               ColName = NumOfClassPerID.columns[(NumOfClassPerID == x[1].values[:-1].max()).any()][0]
                         
               if ColName == NumOfClassPerID['True_Label'].loc[x[0]]:                
                    """
                    If the column (class name), where this highest number of predicted labels belongs to, is the same as the true label, 
                    then the patient has the ovarian cancer subtype CORRECTLY PREDICTED !!
                    """
                    # Adding '1' (each time 1 is a patient) in the diagonal element of the 'true class and correctly predicted class' corresponding entry
                    CM_Patients[ColName].loc[ColName] = np.array(CM_Patients[ColName].loc[ColName]+1)      
                    
               else:     
                    # Adds '1' (each time 1 is a patient) in the POSITION of the falsely predicted class for the givrn true class, in the created confusion matrix,
                    # when the class, where the highest number of predicted image labels per patient belong to, IS NOT the same as the true class for this patient
                    CM_Patients[ColName].loc[NumOfClassPerID['True_Label'].loc[x[0]]] = np.array(CM_Patients[ColName].loc[NumOfClassPerID['True_Label'].loc[x[0]]] +1) 
                
          CM_Patients.fillna(0, inplace=True) 
          
          # Saving the CONFUSION MATRIX ON THE PATIENT LEVEL in the current dataset's results folder
          Project_Dir2=  self.Results_Dirs + '\\' + self.experiment + '_'+ self.network + 'CM_Pnt_' + self.DataType + self.cvtype + str(self.Idx) + '.xlsx'
          CM_Patients.to_excel(Project_Dir2, index= True)
                         
          
          return Predictions,  Predicted_Filtered;   
          