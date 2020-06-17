# -*- coding: utf-8 -*-
"""
         Author: Georgia Liapi, Master Student in Systems Biology, Maastricht University
  Academic Year: 2019-2020
        Project: 'Deep learning-based prediction of molecular subtypes in human ovarian cancer.'
        Utility: Split the primary whole data per set into training and testing
"""

# Import Required Libraries

from collections import Counter
import pandas as pd
import numpy as np
import os
from itertools import chain

# Get the working directory
Current_Dir = os.getcwd()

# Set the tiles (*.jpg) directories for all datasets
OV_SFU_JPG = '..\\SimonFraserUniversityJPG'
OV_DX_JPG  = '..\\TCGA_OV_DX_JPG'
OV_KR_JPG  = '..\\TCGA_OV_KR_JPG'

# Create a list of the jpg images tiles datasets directories
Dirs = [OV_SFU_JPG, OV_DX_JPG, OV_KR_JPG] 

def Test_Data(Tiles_Dir= Dirs, Project_Dir= Current_Dir):
     
     global Final_Matched_DF, Classes
     
     #-------------------------------------------------------------------------
     #           For the Simon Fraser University Ovarian Cancer Dataset 
     #-------------------------------------------------------------------------
     if Tiles_Dir == Dirs[0]:
          
          # Read the final Dataframe-------------------------------------------
          Matched_DF = pd.read_excel(Project_Dir+'\\'+'SimonFraserUniversityJPG_Matched.xlsx' , index_col=0)
          
          # Count the classes--------------------------------------------------
          Classes = list(Matched_DF['Class'].unique())
          print('\n Classes:', Classes)
          
          TilesPerClass = Counter(Matched_DF['Class'])
          print('\n Tiles per Class:', TilesPerClass)
                                      
          # ------------ GENERATE THE TESTDATA --------------------------------
          
          # Count patients per class
          PatientsPerClass = pd.DataFrame()
          
          # For each of the classes in the current dataset
          for idx, i in enumerate(Classes):
               
               # To find the patients' ids per class (ovarian cancer subtype) and save them in the corresponding column number
               PntsPerClass = Matched_DF[Matched_DF['Class']== i].index.unique()
               PatientsPerClass[i] = pd.Series(list(PntsPerClass))
               
          # Count Patients per Class (ascending sorting)-----------------------
          NumOfPatientsPerClass = pd.DataFrame(PatientsPerClass.count().sort_values())
          NumOfPatientsPerClass.columns = ['Patients']
          NumOfPatientsPerClass.to_excel(Project_Dir+'\\'+ os.path.basename(Dirs[0]) + '_Matched_PatientsPerClass.xlsx')
          
          # Class with the less patients
          ClassLessPnts= NumOfPatientsPerClass.iloc[0][0]
          
          # Select the 20% of the total patients, starting from the class with the less patients
          NumOfPnts = np.round(ClassLessPnts * 0.2)
          
          # Define the testdata dataframe--------------------------------------   
          Selected = []
          
          for i in PatientsPerClass.columns:
               NotNaN = PatientsPerClass[i].dropna()
               np.random.seed(3)
               Selected.append(np.random.choice(NotNaN, int(NumOfPnts)))
                    
          # Final Dataframe -leave out the test data---------------------------
          Final = Matched_DF.loc[list(chain.from_iterable(Selected))]
          
          # Count patients per class for the test data
          PatientsPerClass = pd.DataFrame()
          
          # For each of the classes in the current dataset
          for idx, i in enumerate(Classes):
               
               # To find the patients' ids per class (ovarian cancer subtype) and save them in the corresponding column number
               PntsPerClass = Final[Final['Class']== i].index.unique()
               PatientsPerClass[i] = pd.Series(list(PntsPerClass))
               
          # Count Patients per Class for the Test Data(ascending sorting)------
          NumOfPatientsPerClass = pd.DataFrame()     
          NumOfPatientsPerClass = pd.DataFrame(PatientsPerClass.count().sort_values())
          NumOfPatientsPerClass.columns = ['Patients']
          NumOfPatientsPerClass.to_excel(Project_Dir+'\\'+ os.path.basename(Dirs[0]) + '_TestData_PatientsPerClass.xlsx')          
                    
          # Save the testdata
          Final.to_excel(r'{}\\{}.xlsx'.format(Project_Dir,os.path.basename(Tiles_Dir) + '_TestData'), index=True)
          
          # Compare the indices from the primary dataframe and substract those used for testing, to keep the rest for training
          RestIndices = set(Matched_DF.index.unique()) - set(Final.index.unique())
          TrainData = Matched_DF.loc[RestIndices]
          
          # Count patients per class in the Train Data
          PatientsPerClass = pd.DataFrame()
          
          # For each of the classes in the current dataset
          for idx, i in enumerate(Classes):
               
               # To find the patients' ids per class (ovarian cancer subtype) and save them in the corresponding column number
               PntsPerClass = TrainData[TrainData['Class']== i].index.unique()
               PatientsPerClass[i] = pd.Series(list(PntsPerClass))
               
          # Count Patients per Class (ascending sorting)-----------------------
          NumOfPatientsPerClass = pd.DataFrame(PatientsPerClass.count().sort_values())
          NumOfPatientsPerClass.columns = ['Patients']
          NumOfPatientsPerClass.to_excel(Project_Dir+'\\'+ os.path.basename(Dirs[0]) + '_TrainData_PatientsPerClass.xlsx')
          
          TrainData.to_excel(r'{}\\{}.xlsx'.format(Project_Dir,os.path.basename(Tiles_Dir) + '_TrainData'), index=True)
          
          
     #-------------------------------------------------------------------------
     #           For the TCGA-OV-DX-derived image tiles Dataset 
     #-------------------------------------------------------------------------   
     elif Tiles_Dir == Dirs[1]:   
          
          # Read the final Dataframe-------------------------------------------
          Matched_DF = pd.read_excel( Project_Dir+'\\'+'TCGA_OV_DX_JPG_Matched.xlsx', index_col=0)
          
          # Count the classes--------------------------------------------------
          # Instantiate an empty dictionary
          Classes = {}
          
          # For each of the classes in the current dataset
          for elm in list(Matched_DF['Class'].unique()):
               
               # Count patient ids in each class
               Classes[elm] = len(list(Counter(Matched_DF[Matched_DF['Class']==elm].index).keys()))
               
          # Sort the Classes dictionary key-value pairs in a descending order based on the values entries, starting from the class with the highest number of patients 
          Classes = {Y: y for Y, y in sorted(Classes.items(), key=lambda elm: elm[1], reverse=True)}
          print('\n Classes:', Classes)
          
          TilesPerClass = Counter(Matched_DF['Class'])
          print('\n Tiles per Class:', TilesPerClass)
                                      
          # ------------ GENERATE THE TESTDATA --------------------------------
          
          # Count patients per class
          PatientsPerClass = pd.DataFrame()                   
          
          for idx, i in enumerate(Classes):
               
               # The PatientsPerClass table lists all the patient ids per class, starting from the class having highest number of patients
               PntsPerClass = Matched_DF[Matched_DF['Class']== i].index.unique()
               PatientsPerClass[i] = pd.Series(list(PntsPerClass))
               
          # Count Patients per Class (ascending sorting)-----------------------
          NumOfPatientsPerClass = pd.DataFrame()     
          NumOfPatientsPerClass = pd.DataFrame(PatientsPerClass.count().sort_values())
          NumOfPatientsPerClass.columns = ['Patients']
          NumOfPatientsPerClass.to_excel(Project_Dir+'\\'+ os.path.basename(Dirs[1]) + '_Matched_PatientsPerClass.xlsx')
          
          # Class with the less patients
          ClassLessPnts= NumOfPatientsPerClass.iloc[0][0]
          
          # Select the 20% of the total patients, starting from the class with the fewer patients
          NumOfPnts = np.round(ClassLessPnts * 0.2)
          
          # Define the testdata dataframe--------------------------------------  
          Selected = []
          
          for i in PatientsPerClass.columns:
               NotNaN = PatientsPerClass[i].dropna()
               np.random.seed(3)
               Selected.append(np.random.choice(NotNaN, int(NumOfPnts)))
                    
          # Final Dataframe -leave out the test data---------------------------
          Final = Matched_DF.loc[list(chain.from_iterable(Selected))]
                    
          # Save the testdata
          Final.to_excel(r'{}\\{}.xlsx'.format(Project_Dir,os.path.basename(Tiles_Dir) + '_TestData'), index=True)
          
          # Count patients per class for the test data
          PatientsPerClass = pd.DataFrame()
          
          for idx, i in enumerate(Classes):
                    
               # The PatientsPerClass table lists all the patient ids per class, starting from the class having highest number of patients
               PntsPerClass = Final[Final['Class']== i].index.unique()
               PatientsPerClass[i] = pd.Series(list(PntsPerClass))
               
          # Count Patients per Class for the Test Data(ascending sorting)------
          NumOfPatientsPerClass =pd.DataFrame()     
          NumOfPatientsPerClass = pd.DataFrame(PatientsPerClass.count().sort_values())
          NumOfPatientsPerClass.columns = ['Patients']
          NumOfPatientsPerClass.to_excel(Project_Dir+'\\'+ os.path.basename(Dirs[1]) + '_TestData_PatientsPerClass.xlsx')
          
          # Compare the indices from the primary dataframe and substract those used for testing, to keep the rest for training
          RestIndices = set(Matched_DF.index.unique()) - set(Final.index.unique())
          TrainData = Matched_DF.loc[RestIndices]
          
          # Count patients per class in the Train Data------------------------
          PatientsPerClass = pd.DataFrame()
          
          for idx, i in enumerate(Classes):
               
               # The PatientsPerClass table lists all the patient ids per class, starting from the class having highest number of patients
               PntsPerClass = TrainData[TrainData['Class']== i].index.unique()
               PatientsPerClass[i] = pd.Series(list(PntsPerClass))
               
          # Count Patients per Class (ascending sorting)-----------------------
          NumOfPatientsPerClass = pd.DataFrame(PatientsPerClass.count().sort_values())
          NumOfPatientsPerClass.columns = ['Patients']
          NumOfPatientsPerClass.to_excel(Project_Dir+'\\'+ os.path.basename(Dirs[1]) + '_TrainData_PatientsPerClass.xlsx')
          
          TrainData.to_excel(r'{}\\{}.xlsx'.format(Project_Dir,os.path.basename(Tiles_Dir) + '_TrainData'), index=True)


     #-------------------------------------------------------------------------
     #            For the TCGA-OV-KR-derived image tiles Dataset 
     #-------------------------------------------------------------------------    
     elif Tiles_Dir == Dirs[2]:   
          
          # Read the final Dataframe------------------------------------------------
          Matched_DF = pd.read_excel( Project_Dir+'\\'+'TCGA_OV_KR_JPG_Matched.xlsx', index_col=0)
          
          # Instantiate an empty dictionary
          Classes = {}
         
          # For each of the classes in the current dataset
          for elm in list(Matched_DF['Class'].unique()):
               
               # Count patient ids in each class
               Classes[elm] = len(list(Counter(Matched_DF[Matched_DF['Class']==elm].index).keys()))
          
          # Sort the Classes dictionary key-value pairs in a descending order based on the values entries, starting from the class with the highest number of patients  
          Classes = {Y: y for Y, y in sorted(Classes.items(), key=lambda elm: elm[1], reverse=True)}
          print('\n Classes:', Classes)
          
          TilesPerClass = Counter(Matched_DF['Class'])
          print('\n Tiles per Class:', TilesPerClass)
                                                                            
          # ------------ GENERATE THE TESTDATA --------------------------------------
          
          # Count patients per class
          PatientsPerClass = pd.DataFrame()
          
          for idx, i in enumerate(Classes):
                
               # The PatientsPerClass table lists all the patient ids per class, starting from the class having highest number of patients 
               PntsPerClass = Matched_DF[Matched_DF['Class']== i].index.unique()
               PatientsPerClass[i] = pd.Series(list(PntsPerClass))
               
          # Count Patients per Class (ascending sorting)-----------------------------
          NumOfPatientsPerClass = pd.DataFrame()     
          NumOfPatientsPerClass = pd.DataFrame(PatientsPerClass.count().sort_values())
          NumOfPatientsPerClass.columns = ['Patients']
          NumOfPatientsPerClass.to_excel(Project_Dir+'\\'+ os.path.basename(Dirs[2]) + '_Matched_PatientsPerClass.xlsx')
          
          # Class with the less patients
          ClassLessPnts= NumOfPatientsPerClass.iloc[0][0]
          
          # Select the 20% of the total patients, starting from the class with the less patients
          NumOfPnts = np.round(ClassLessPnts * 0.2)
          
          # Define the testdata dataframe------------------------------------------     
          Selected = []
          
          for i in PatientsPerClass.columns:
               NotNaN = PatientsPerClass[i].dropna()
               np.random.seed(3)
               Selected.append(np.random.choice(NotNaN, int(NumOfPnts)))
                    
          # Final Dataframe -leave out the test data--------------------------------
          Final = Matched_DF.loc[list(chain.from_iterable(Selected))]
     
          # Save the testdata
          Final.to_excel(r'{}\\{}.xlsx'.format(Project_Dir,os.path.basename(Tiles_Dir) + '_TestData'), index=True)
          
          # Count patients per class for the test data
          PatientsPerClass = pd.DataFrame()
          
          for idx, i in enumerate(Classes):
               
               # The PatientsPerClass table lists all the patient ids per class, starting from the class having highest number of patients
               PntsPerClass = Final[Final['Class']== i].index.unique()
               PatientsPerClass[i] = pd.Series(list(PntsPerClass))
               
          # Count Patients per Class for the Test Data(ascending sorting)-----------------------------
          NumOfPatientsPerClass =pd.DataFrame()     
          NumOfPatientsPerClass = pd.DataFrame(PatientsPerClass.count().sort_values())
          NumOfPatientsPerClass.columns = ['Patients']
          NumOfPatientsPerClass.to_excel(Project_Dir+'\\'+ os.path.basename(Dirs[2]) + '_TestData_PatientsPerClass.xlsx')
          
          # Compare the indices from the primary dataframe and substract those used for testing, to keep the rest for training
          RestIndices = set(Matched_DF.index.unique()) - set(Final.index.unique())
          TrainData = Matched_DF.loc[RestIndices]
          
          # Count patients per class in the Train Data
          PatientsPerClass = pd.DataFrame()
          
          for idx, i in enumerate(Classes):
               
               # The PatientsPerClass table lists all the patient ids per class, starting from the class having highest number of patients
               PntsPerClass = TrainData[TrainData['Class']== i].index.unique()
               PatientsPerClass[i] = pd.Series(list(PntsPerClass))
               
          # Count Patients per Class (ascending sorting)-----------------------------
          NumOfPatientsPerClass = pd.DataFrame(PatientsPerClass.count().sort_values())
          NumOfPatientsPerClass.columns = ['Patients']
          NumOfPatientsPerClass.to_excel(Project_Dir+'\\'+ os.path.basename(Dirs[2]) + '_TrainData_PatientsPerClass.xlsx')
          
          TrainData.to_excel(r'{}\\{}.xlsx'.format(Project_Dir,os.path.basename(Tiles_Dir) + '_TrainData'), index=True)     
          
          return Final, TrainData;
