# -*- coding: utf-8 -*-
"""
          Author: Georgia Liapi, Master Student in Systems Biology, Maastricht University
   Academic Year: 2019-2020
         Project: 'Deep learning-based prediction of molecular subtypes in human ovarian cancer.'
         Utility: Loads the whole training data excel for the selected dataset and generates folds of data on a patient level
                  
            NOTE: The condition for generating image tiles data folds on the patient level is:
                  Image tiles from one specific patient are present only in one of the k-folds.
                  
                  There is a checking point in the following script, for each selected type of k-fold cross validation,
                  where, after folds are created, is it checked if images from one patient exist in more than one folds; the results are printed in the console in real time.
                        
"""

# Import Required Libraries
from collections import Counter, OrderedDict
import pandas as pd
import numpy as np
import os, operator
from itertools import chain

# Here, I do not call the Match_Patients_Classes() function because it is taking a long time to let this run. I am importing the dataframe that has been earlier extracted from 
# the Match_Patients_Classes(), instead.

# Get the working directory
Current_Dir = os.getcwd()

# Set the image tiles (*.jpg) directories for all datasets
OV_SFU_JPG = '..\\SimonFraserUniversityJPG'
OV_DX_JPG  = '..\\TCGA_OV_DX_JPG'
OV_KR_JPG  = '..\\TCGA_OV_KR_JPG'

# Create a list of the directories
Dirs = [OV_SFU_JPG, OV_DX_JPG, OV_KR_JPG] 


# Define a function to generate k-folds 
def k_folds(Tiles_Dir= Dirs, Project_Dir= Current_Dir, folds=3):
     
     global Train_DF, Classes
          
     #  Importing the whole training image tiles data for the selected dataset   
     if Tiles_Dir == Dirs[0]:
          
          # Read the final Dataframe------------------------------------------------
          Train_DF = pd.read_excel(Project_Dir+'\\' + 'SimonFraserUniversityJPG_TrainData.xlsx', index_col=0)
          
     elif Tiles_Dir == Dirs[1]:
          
          # Read the final Dataframe------------------------------------------------
          Train_DF = pd.read_excel(Project_Dir+'\\' + 'TCGA_OV_DX_JPG_TrainData.xlsx', index_col=0)    
          
     elif Tiles_Dir == Dirs[2]:
          
          # Read the final Dataframe------------------------------------------------
          Train_DF = pd.read_excel(Project_Dir+'\\' + 'TCGA_OV_KR_JPG_TrainData.xlsx', index_col=0)         
          
     #-------------------------------------------------------------------------     
     #       F O R    T H E    S E L E C T E D    D A T A S E T     
     # ------------------------------------------------------------------------
          
     # Count the classes (ovarian cancer subtypes)
     Classes = list(Train_DF['Class'].unique())
     print('\n Classes:', Classes)
     
     # Count the image tiles per Class
     TilesPerClass = Counter(Train_DF['Class'])
     print('\n Tiles per Class:', TilesPerClass)
     
     # Find the ids of the Patients Per Class 
     Dict = {}
     for rp in range(len(Classes)):
          for els in Classes:         
              Dict[els]=len(Train_DF[Train_DF['Class']==els].index.unique())
     
     #  Sort keys (Class Names) based on descending values
     # Eg. {'High-grade serous': 28, 'Clear cell': 18, 'Endometrioid': 9, 'Mucinous': 8, 'Low-grade serous': 7}
     
     Ordered_Dict = {Y: y for Y, y in sorted(Dict.items(), key=lambda elm: elm[1], reverse=True)}
         
     # Instantiate a dataframe
     PatientsPerClass = pd.DataFrame(columns= list(Ordered_Dict.keys()))
         
     # Expand all columns in a printed pd
     pd.set_option('display.max_columns', 3), pd.set_option('display.expand_frame_repr', False)
     pd.set_option('max_colwidth', 3)
     
     # Provide all patient IDs for each class in the current dataset
     for idx, i in enumerate(list(Ordered_Dict.keys())):
               
          PntsPerClass = Train_DF[Train_DF['Class']== i].index.unique()
          PatientsPerClass[i]= pd.Series(PntsPerClass)
         
     # Count Patients per Class (ascending sorting)
     NumOfPatientsPerClass = PatientsPerClass.count().sort_values()
     
     # Class with the smallest number of patients
     ClassWithLessPnts= NumOfPatientsPerClass[0]
     print(' \n\n The class with the less patients({}) is "{}" \n\n'.format(ClassWithLessPnts, NumOfPatientsPerClass.keys()[0]))  
     
     # ------------------------------------------------------------------------
     #                             GENERATE FOLDS
     # ------------------------------------------------------------------------
     # *************** For 3-fold cross validation ***********************
     
     if folds==3:
          
          #Instantiate the dataframes
          Fold_1, Fold_2, Fold_3 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
          SelectedPatientF1 = []
          SelectedPatientF2 = []
          SelectedPatientF3 = []
          Common = pd.DataFrame()
          
          # Define the folds **************************************************
          
          # For Fold_1_________________________________________________________  
          
          # Selecting the indices (patient ids) for Fold_1                    
          for els1 in list(Ordered_Dict.keys()):
               
               # Set a seed for reproducibility
               # The following it to select patients from each class (classes are in the Dict keys) to use for the current fold
               np.random.seed(1)
               SelectedPatientF1.append(list(np.random.choice(list(set(PatientsPerClass[els1]) - set(PatientsPerClass[els1][pd.isnull(PatientsPerClass[els1])])),
                                                   round((Ordered_Dict[els1]*0.85)/2))))
           
          Fold_1 = Train_DF.loc[list(chain(*SelectedPatientF1))]
               
          Fold_1_TilesPerClass = Counter(Fold_1['Class'])
          print(' \n Tiles per Class in Fold_1: \n\{}'.format(Fold_1_TilesPerClass))
               
               
          # For Fold_2_________________________________________________________
          
          # Find the rest of the patient ids
          PatientsPerClassΝew = PatientsPerClass
          PatientsPerClassΝew = list(set(list(chain(*PatientsPerClassΝew.values))) - set(list(chain(*SelectedPatientF1))))
          
          # Remove any nan values
          for idx, items in enumerate(PatientsPerClassΝew):
               if pd.isnull(items) == True :
                  PatientsPerClassΝew.remove(items)
          print(' \n\n The rest of the indices after preparing Fold_1:', PatientsPerClassΝew)
         
          # Selecting the indices for Fold_2
          for Elm in list(Ordered_Dict.keys()):
               
               np.random.seed(1)
               
               # Compare which of the rest IDs for this class are in the primary dataframe, and filter them out
               Common[Elm] =  pd.Series(list(set(list(Train_DF[Train_DF['Class']== Elm].index.unique())) & set(list(PatientsPerClassΝew))))
              
               SelectedPatientF2.append(list(np.random.choice(list(set(Common[Elm]) - set(Common[Elm][pd.isnull(Common[Elm])])),
                                                   round((Ordered_Dict[Elm]*0.85)/2))))
           
          Fold_2 = Train_DF.loc[list(chain(*SelectedPatientF2))]
               
          Fold_2_TilesPerClass = Counter(Fold_2['Class'])
          print(' \nTiles per Class in Fold_2: \n\n {}'.format(Fold_2_TilesPerClass))          
           
          # For Fold_3_________________________________________________________
          
          # Find the remaining patient ids (not used yet)
          Used_ids = []
          Used_ids.append(list(chain(*SelectedPatientF2)))
          Used_ids.append(list(chain(*SelectedPatientF1)))
          
          # Unnest lists of lists
          Used_ids = list(chain(*Used_ids))
          SelectedPatientF3 = list(set(list(chain(*PatientsPerClass.values))) - set(Used_ids))
          SelectedPatientF3 = list(set(SelectedPatientF3) - set(list(pd.Series(SelectedPatientF3)[pd.isnull(SelectedPatientF3)])))
          
          Fold_3 = Train_DF.loc[SelectedPatientF3]
          
          Fold_3_TilesPerClass = Counter(Fold_3['Class'])
          print(" \n Tiles per Class in Fold_3: \n\n {}".format(Fold_3_TilesPerClass))          
          
          # Check if one patient's images exist in more than one folds
          thesame0 = list(set(list(Fold_1.index.unique())) & set(list(Fold_2.index.unique())))
          thesame1 = list(set(list(Fold_1.index.unique())) & set(list(Fold_3.index.unique())))
          thesame2 = list(set(list(Fold_2.index.unique())) & set(list(Fold_3.index.unique())))
          
          if thesame0== [] and thesame1== [] and thesame2== [] :
               print(" \n\n This is a check to see if a patient exists in more than one folds: \n\n Fold_1 and Fold_2-> Common: {} \
                     \n Fold_1 and Fold_3-> Common: {} \n Fold_2 and Fold_3-> Common: {}".format('NAN','NAN','NAN'))
          else:
               print("This is a check to see if a patient exists in more than one folds: \n\n Fold_1 and Fold_2-> Common: {} \
                     \n Fold_1 and Fold_3-> Common: {} \n Fold_2 and Fold_3-> Common: {}".format(thesame0,thesame1,thesame2))      
         
          Fold_1_TilesPerPatient = Counter(Fold_1.index)       
          Fold_2_TilesPerPatient = Counter(Fold_2.index)  
          Fold_3_TilesPerPatient = Counter(Fold_3.index)  
          
          print(' \n Tiles per Patient and Fold: \n\n Fold_1: {} \n\n Fold_2: {} \n\n Fold_3: {} '.format(Fold_1_TilesPerPatient, Fold_2_TilesPerPatient,  Fold_3_TilesPerPatient))
          
        
          # Under-sampling  ********************************************************
          """
          DESIRED RE-SAMPLING RESULT: 
               For the image tiles representing a class within a fold, resample the images taken from each patient, 
               by re-taking (randomly) from each patient as many images as the average number of image tiles from all 
               patients in all classes and all folds.
          """
          
          # Aggregating the total number of image tiles per class from all folds.
          TilesPerPatientAllFolds = list(Fold_1_TilesPerPatient.values()), list(Fold_2_TilesPerPatient.values()), list(Fold_3_TilesPerPatient.values())
     
          # Sort this list in an ascending order and use the first element (smallest)
          TilesPerPatientAllFolds =  sorted(list(set(list(chain(* TilesPerPatientAllFolds)))))
          
          # Finding the average number of image tiles from all patients, in all classes and all folds
          MeanNumOfTilesPerPatient = int(np.array( TilesPerPatientAllFolds).mean())
          
          # Create a list of the Folds
          Folds = [Fold_1, Fold_2, Fold_3]
          fold_1 = pd.DataFrame()
          fold_2 = pd.DataFrame()
          fold_3 = pd.DataFrame()
          
          # Under-sample the image tiles per patient, based on the above-mentioned average number
          # If a patient's number of images cannot fullfill the average, take as many images as this patient currently has
          for fold in Folds:
                             
               if fold is Folds[0]:
                    
                     for Index in list(Fold_1.index.unique()):
                         
                         if len(Fold_1[Fold_1.index == Index]) > MeanNumOfTilesPerPatient:
                              
                               fold_1 = fold_1.append(Fold_1[Fold_1.index== Index].sample(n = MeanNumOfTilesPerPatient, replace=False, random_state=1))
                         else: 
                              
                               fold_1 = fold_1.append(Fold_1[Fold_1.index== Index])
                              
               elif fold is Folds[1]: 
                    
                   for Index in list(Fold_2.index.unique()):
                         
                         if len(Fold_2[Fold_2.index == Index]) > MeanNumOfTilesPerPatient:
                              
                               fold_2 = fold_2.append(Fold_2[Fold_2.index== Index].sample(n = MeanNumOfTilesPerPatient, replace=False, random_state=1))
                         else: 
                              
                               fold_2 = fold_2.append(Fold_2[Fold_2.index== Index])
               else:
                    
                     for Index in list(Fold_3.index.unique()):
                         
                         if len(Fold_3[Fold_3.index == Index]) > MeanNumOfTilesPerPatient:
                              
                               fold_3 = fold_3.append(Fold_3[Fold_3.index== Index].sample(n = MeanNumOfTilesPerPatient, replace=False, random_state=1))
                         else: 
                              
                               fold_3 = fold_3.append(Fold_3[Fold_3.index== Index])
     
          # AFTER UNDER-SAMPLING ********************************************************
          fold_1_TilesPerClass = Counter(fold_1['Class'])       
          fold_2_TilesPerClass = Counter(fold_2['Class'])  
          fold_3_TilesPerClass = Counter(fold_3['Class'])  
          
          print(' \n\n Tiles per Class after ** UNDER-SAMPLING ** : \n\n fold_1: {} \n\n fold_2: {} \n\n fold_3: {} '.format(fold_1_TilesPerClass, fold_2_TilesPerClass,  fold_3_TilesPerClass))
          
          # Save the folds, with the fold number and the dataset name          
          fold_1.to_excel(r'{}\\{}.xlsx'.format(Project_Dir,os.path.basename(Tiles_Dir) + '_fold_1'), index=True)       
          fold_2.to_excel(r'{}\\{}.xlsx'.format(Project_Dir,os.path.basename(Tiles_Dir) + '_fold_2'), index=True)
          fold_3.to_excel(r'{}\\{}.xlsx'.format(Project_Dir,os.path.basename(Tiles_Dir) + '_fold_3'), index=True)          
               
          # Create a list of the folds
          folds = [fold_1, fold_2, fold_3]
     
     # ******************** For 5-fold cross validation ***********************     
     elif folds==5:   
          
          #Instantiate the dataframes
          Fold_1, Fold_2, Fold_3, Fold_4, Fold_5 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
          SelectedPatientF1 = []
          SelectedPatientF2 = []
          SelectedPatientF3 = []
          SelectedPatientF4 = []
          SelectedPatientF5 = []
          
          Common = pd.DataFrame()
          
          # Define the folds **************************************************
          
          # For Fold_1_________________________________________________________  
          
          # Selecting the indices for Fold_1                    
          for els1 in list(Ordered_Dict.keys()):
               
               # Set a seed for reproducibility
               # The following is to select patients from each class (classes are in the Dict keys) to use for the current fold
               np.random.seed(1)
               
               # The following division rule serves to split the number of patients between the folds, as much equally as possible
               SelectedPatientF1.append(list(np.random.choice(list(set(PatientsPerClass[els1]) - set(PatientsPerClass[els1][pd.isnull(PatientsPerClass[els1])])),
                                                   round((Ordered_Dict[els1]*0.85)/4))))
           
          Fold_1 = Train_DF.loc[list(chain(*SelectedPatientF1))]
               
          Fold_1_TilesPerClass = Counter(Fold_1['Class'])
          print(' \n Tiles per Class in Fold_1: \n\n {}'.format(Fold_1_TilesPerClass))
               
               
          # For Fold_2_________________________________________________________
          
          # Find the rest of the patient ids
          PatientsPerClassΝew = PatientsPerClass
          PatientsPerClassΝew = list(set(list(chain(*PatientsPerClassΝew.values))) - set(list(chain(*SelectedPatientF1))))
          
          # Remove any nan values
          for idx, items in enumerate(PatientsPerClassΝew):
               if pd.isnull(items) == True :
                  PatientsPerClassΝew.remove(items)
          print(' \n\n The rest of the indices after preparing Fold_1:', PatientsPerClassΝew)
         
          # Selecting the indices for Fold_2
          for Elm in list(Ordered_Dict.keys()):
               
               np.random.seed(1)
               
               # Compare which of the rest of the patient IDs for each class are in the primary dataframe, and filter them out
               Common[Elm] =  pd.Series(list(set(list(Train_DF[Train_DF['Class']== Elm].index.unique())) & set(list(PatientsPerClassΝew))))
              
               # The following division rule serves to split the number of patients between the folds, as much equally as possible
               SelectedPatientF2.append(list(np.random.choice(list(set(Common[Elm]) - set(Common[Elm][pd.isnull(Common[Elm])])),
                                                   round((Ordered_Dict[Elm]*0.85)/4))))
           
          Fold_2 = Train_DF.loc[list(chain(*SelectedPatientF2))]
               
          Fold_2_TilesPerClass = Counter(Fold_2['Class'])
          print(' \nTiles per Class in Fold_2: \n\n {}'.format(Fold_2_TilesPerClass))          
           
          
          # For Fold_3_________________________________________________________
          
          # Find the rest of the patient ids
          PatientsPerClassΝew = PatientsPerClass
          PatientsPerClassΝew = list(set(list(chain(*PatientsPerClassΝew.values))) - set(list(chain(*SelectedPatientF1+SelectedPatientF2))))
          
          # Remove any nan values
          for idx, items in enumerate(PatientsPerClassΝew):
               if pd.isnull(items) == True :
                  PatientsPerClassΝew.remove(items)
          print('The rest of the indices after preparing Fold_2:', PatientsPerClassΝew)
         
          # Selecting the indices for Fold_3
          for Elm in list(Ordered_Dict.keys()):
               
               np.random.seed(1)
               
               # Compare which of the rest IDs for this class are in the primary dataframe, and filter them out
               Common[Elm] =  pd.Series(list(set(list(Train_DF[Train_DF['Class']== Elm].index.unique())) & set(list(PatientsPerClassΝew))))
              
               # The following division rule serves to split the number of patients between the folds, as much equally as possible
               SelectedPatientF3.append(list(np.random.choice(list(set(Common[Elm]) - set(Common[Elm][pd.isnull(Common[Elm])])),
                                                   round((Ordered_Dict[Elm]*0.85)/4))))
           
          Fold_3 = Train_DF.loc[list(chain(*SelectedPatientF3))]
               
          Fold_3_TilesPerClass = Counter(Fold_3['Class'])
          print(' \nTiles per Class in Fold_3: \n\n {}'.format(Fold_3_TilesPerClass))          
           
          
          # For Fold_4_________________________________________________________
          
          # Find the rest of the patient ids
          PatientsPerClassΝew = PatientsPerClass
          PatientsPerClassΝew = list(set(list(chain(*PatientsPerClassΝew.values))) - set(list(chain(*SelectedPatientF1+SelectedPatientF2+SelectedPatientF3))))
          
          # Remove any nan values
          for idx, items in enumerate(PatientsPerClassΝew):
               if pd.isnull(items) == True :
                  PatientsPerClassΝew.remove(items)
          print('The rest of the indices after preparing Fold_3:', PatientsPerClassΝew)
         
          # Selecting the indices for Fold_3
          for Elm in list(Ordered_Dict.keys()):
               
               np.random.seed(1)
               
               # Compare which of the rest IDs for this class are in the primary dataframe, and filter them out
               Common[Elm] =  pd.Series(list(set(list(Train_DF[Train_DF['Class']== Elm].index.unique())) & set(list(PatientsPerClassΝew))))
              
               # The following division rule serves to split the number of patients between the folds, as much equally as possible
               SelectedPatientF4.append(list(np.random.choice(list(set(Common[Elm]) - set(Common[Elm][pd.isnull(Common[Elm])])),
                                                   round((Ordered_Dict[Elm]*0.85)/4))))
           
          Fold_4 = Train_DF.loc[list(chain(*SelectedPatientF4))]
               
          Fold_4_TilesPerClass = Counter(Fold_4['Class'])
          print(' \nTiles per Class in Fold_4: \n\n {}'.format(Fold_4_TilesPerClass))  
          
          # For Fold_5_________________________________________________________
          # Find the remaining patient ids (not used yet)
          Used_ids = []
          Used_ids.append(list(chain(*SelectedPatientF1)))
          Used_ids.append(list(chain(*SelectedPatientF2)))
          Used_ids.append(list(chain(*SelectedPatientF3)))
          Used_ids.append(list(chain(*SelectedPatientF4)))
          
          
          # Unnest lists of lists
          Used_ids = list(chain(*Used_ids))
          SelectedPatientF5 = list(set(list(chain(*PatientsPerClass.values))) - set(Used_ids))
          SelectedPatientF5 = list(set(SelectedPatientF5) - set(list(pd.Series(SelectedPatientF5)[pd.isnull(SelectedPatientF5)])))
          
          # Generate the last fold
          Fold_5 = Train_DF.loc[SelectedPatientF5]
          
          # Check if one patient exists in more than one folds
          thesame0 = list(set(list(Fold_1.index.unique())) & set(list(Fold_2.index.unique())))
          thesame1 = list(set(list(Fold_1.index.unique())) & set(list(Fold_3.index.unique())))
          thesame2 = list(set(list(Fold_1.index.unique())) & set(list(Fold_4.index.unique())))
          thesame3 = list(set(list(Fold_1.index.unique())) & set(list(Fold_5.index.unique())))
          
          thesame4 = list(set(list(Fold_2.index.unique())) & set(list(Fold_3.index.unique())))
          thesame5 = list(set(list(Fold_2.index.unique())) & set(list(Fold_4.index.unique())))
          thesame6 = list(set(list(Fold_2.index.unique())) & set(list(Fold_5.index.unique())))
          
          thesame7 = list(set(list(Fold_4.index.unique())) & set(list(Fold_5.index.unique())))
          thesame8 = list(set(list(Fold_3.index.unique())) & set(list(Fold_4.index.unique())))
          thesame9 = list(set(list(Fold_3.index.unique())) & set(list(Fold_5.index.unique())))
          
          
          if thesame0== [] and thesame1== [] and thesame2== [] and thesame3== [] and \
             thesame4== [] and thesame5== [] and thesame6== [] and thesame7== [] and thesame8== [] and thesame9== []:
               
               print("This is a check to see if a patient exists in more than one folds: \n\n Fold_1 and Fold_2-> Common: {} \
                     \n Fold_1 and Fold_3-> Common: {} \n Fold_1 and Fold_4-> Common: {} \n Fold_1 and Fold_5-> Common: {} \n\n Fold_2 and Fold_3-> Common: {} \
                     \n Fold_3 and Fold_4-> Common: {} \n Fold_2 and Fold_5-> Common: {} \n\n Fold_4 and Fold_5-> Common: {} \
                     \n Fold_3 and Fold_4-> Common: {} \n Fold_3 and Fold_5-> Common: {}".format('NAN','NAN','NAN', 'NAN','NAN','NAN', 'NAN','NAN','NAN', 'NAN' ))
          else:
               print("This is a check to see if a patient exists in more than one folds: \n\n Fold_1 and Fold_2-> Common: {} \
                     \n Fold_1 and Fold_3-> Common: {} \n Fold_1 and Fold_4-> Common: {} \n Fold_1 and Fold_5-> Common: {} \n\n Fold_2 and Fold_3-> Common: {} \
                     \n Fold_3 and Fold_4-> Common: {} \n Fold_2 and Fold_5-> Common: {} \n\n Fold_4 and Fold_5-> Common: {} \
                     \n Fold_3 and Fold_4-> Common: {} \n Fold_3 and Fold_5-> Common: {}".format(thesame0,thesame1,thesame2, thesame3,thesame4,thesame5, thesame6,thesame7,thesame8, thesame9))      
         
          Fold_1_TilesPerPatientAllFolds = OrderedDict(Counter(Fold_1.index))       
          Fold_2_TilesPerPatientAllFolds = OrderedDict(Counter(Fold_2.index))  
          Fold_3_TilesPerPatientAllFolds = OrderedDict(Counter(Fold_3.index)) 
          Fold_4_TilesPerPatientAllFolds = OrderedDict(Counter(Fold_4.index))       
          Fold_5_TilesPerPatientAllFolds = OrderedDict(Counter(Fold_5.index))
          print(' \n Tiles per Patient and Fold: \n\n Fold_1: {} \n\n Fold_2: {} \n\n Fold_3: {} \n\n Fold_4: {} \n\n Fold_5: {}'.format(Fold_1_TilesPerPatientAllFolds, \
                                                                                                                              Fold_2_TilesPerPatientAllFolds,  Fold_3_TilesPerPatientAllFolds,\
                                                                                                                              Fold_4_TilesPerPatientAllFolds, Fold_5_TilesPerPatientAllFolds))
          
          # Undersampling  ********************************************************
          """
          DESIRED RE-SAMPLING RESULT: 
               For the image tiles representing a class within a fold, resample the images taken from each patient, 
               by re-taking (randomly) from each patient as many images as the average number of image tiles from all 
               patients in all classes and all folds.
          """
          
          # Aggregating the total number of image tiles per class from all folds.
          TilesPerPatientAllFolds = list(Fold_1_TilesPerPatientAllFolds.values()), list(Fold_2_TilesPerPatientAllFolds.values()), list(Fold_3_TilesPerPatientAllFolds.values()),\
                                    list(Fold_4_TilesPerPatientAllFolds.values()), list(Fold_5_TilesPerPatientAllFolds.values())
     
          # Sort this list in an ascending order and use the first element (smallest)
          TilesPerPatientAllFolds =  sorted(list(set(list(chain(*TilesPerPatientAllFolds)))))
          MeanNumOfTilesPerPatient = int(np.array(TilesPerPatientAllFolds).mean())
          
          # Create a list of the Folds
          Folds = [Fold_1, Fold_2, Fold_3, Fold_4, Fold_5]
          fold_1 = pd.DataFrame()
          fold_2 = pd.DataFrame()
          fold_3 = pd.DataFrame()
          fold_4 = pd.DataFrame()
          fold_5 = pd.DataFrame()
          
          # Under-sample the image tiles per patient, based on the above-mentioned average number
          # If a patient's number of images cannot fullfill the average, take as many images as this patient currently has
          for fold in Folds:
               
               if fold is Folds[0]:
                    
                     for Index in list(Fold_1.index.unique()):
                         
                         if len(Fold_1[Fold_1.index == Index]) >   MeanNumOfTilesPerPatient:
                              
                               fold_1 = fold_1.append(Fold_1[Fold_1.index== Index].sample(n = MeanNumOfTilesPerPatient, replace=False, random_state=1))
                         else: 
                              
                               fold_1 = fold_1.append(Fold_1[Fold_1.index== Index])
                              
               elif fold is Folds[1]: 
                    
                   for Index in list(Fold_2.index.unique()):
                         
                         if len(Fold_2[Fold_2.index == Index]) >   MeanNumOfTilesPerPatient:
                              
                               fold_2 = fold_2.append(Fold_2[Fold_2.index== Index].sample(n = MeanNumOfTilesPerPatient, replace=False, random_state=1))
                         else: 
                              
                               fold_2 = fold_2.append(Fold_2[Fold_2.index== Index])
                              
               elif fold is Folds[2]: 
                    
                     for Index in list(Fold_3.index.unique()):
                         
                         if len(Fold_3[Fold_3.index == Index]) >   MeanNumOfTilesPerPatient:
                              
                               fold_3 = fold_3.append(Fold_3[Fold_3.index== Index].sample(n = MeanNumOfTilesPerPatient, replace=False, random_state=1))
                         else: 
                              
                               fold_3 = fold_3.append(Fold_3[Fold_3.index== Index])
                              
               elif fold is Folds[3]: 
               
                     for Index in list(Fold_4.index.unique()):
                         
                         if len(Fold_4[Fold_4.index == Index]) >   MeanNumOfTilesPerPatient:
                              
                               fold_4 = fold_4.append(Fold_4[Fold_4.index== Index].sample(n = MeanNumOfTilesPerPatient, replace=False, random_state=1))
                         else: 
                              
                               fold_4 = fold_4.append(Fold_4[Fold_4.index== Index])         
                              
               elif fold is Folds[4]: 
               
                     for Index in list(Fold_5.index.unique()):
                         
                         if len(Fold_5[Fold_5.index == Index]) >   MeanNumOfTilesPerPatient:
                              
                               fold_5 = fold_5.append(Fold_5[Fold_5.index== Index].sample(n = MeanNumOfTilesPerPatient, replace=False, random_state=1))
                         else: 
                              
                               fold_5 = fold_5.append(Fold_5[Fold_5.index== Index])                     
     
          # AFTER UNDER-SAMPLING ********************************************************
          fold_1_TilesPerClass = Counter(fold_1['Class'])       
          fold_2_TilesPerClass = Counter(fold_2['Class'])  
          fold_3_TilesPerClass = Counter(fold_3['Class'])  
          fold_4_TilesPerClass = Counter(fold_4['Class'])  
          fold_5_TilesPerClass = Counter(fold_5['Class'])  
          
          print('Tiles per Class after ** UNDER-SAMPLING ** : \n\n fold_1: {} \n\n fold_2: {} \n\n fold_3: {} \n\n fold_4: {} \n\n fold_5: {}'.format(\
             fold_1_TilesPerClass, fold_2_TilesPerClass,  fold_3_TilesPerClass, fold_4_TilesPerClass,  fold_5_TilesPerClass))
          
          # Save the folds, with the fold number and the dataset name  ('5f' distinguishes designates the 5-fold cross validation tables in the PROJECT\SCRIPTS folder)               
          fold_1.to_excel(r'{}\\{}.xlsx'.format(Project_Dir,os.path.basename(Tiles_Dir) + '_5f_fold_1'), index=True)       
          fold_2.to_excel(r'{}\\{}.xlsx'.format(Project_Dir,os.path.basename(Tiles_Dir) + '_5f_fold_2'), index=True)
          fold_3.to_excel(r'{}\\{}.xlsx'.format(Project_Dir,os.path.basename(Tiles_Dir) + '_5f_fold_3'), index=True)          
          fold_4.to_excel(r'{}\\{}.xlsx'.format(Project_Dir,os.path.basename(Tiles_Dir) + '_5f_fold_4'), index=True)
          fold_5.to_excel(r'{}\\{}.xlsx'.format(Project_Dir,os.path.basename(Tiles_Dir) + '_5f_fold_5'), index=True)          
               
          # Create a list of the folds
          folds = [fold_1, fold_2, fold_3, fold_4, fold_5]
          return folds;






