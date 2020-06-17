# -*- coding: utf-8 -*-
"""
            Author: Georgia Liapi, Master Student in Systems Biology, Maastricht University
     Academic Year: 2019-2020
           Project: 'Deep learning-based prediction of molecular subtypes in human ovarian cancer'
           Utility: Loads the image tiles data for the selected dataset on the Train_Val_Pread.py or the Train_whole_Test.py function
                    If folds='None' and state='train', the whole training data for the selected dataset are imported
                    If folds= 3 or 5 and state='train', the given folds for the selected dataset are imported
                    If folds='None' and state='test', the final testing data for the selected dataset are imported
"""

# Import the necessary libraries
import pandas as pd
import glob, os


# Get the working directory
Current_Dir = os.getcwd()

# Set the tiles (*.jpg) directories for all datasets
OV_SFU_JPG = '..\\SimonFraserUniversityJPG'
OV_DX_JPG  = '..\\TCGA_OV_DX_JPG'
OV_KR_JPG  = '..\\TCGA_OV_KR_JPG'
Dirs = [OV_SFU_JPG, OV_DX_JPG, OV_KR_JPG] 


class DataGen(object):
     
     def __init__(self, Tiles_Dir, folds, state='train', Project_Dir = Current_Dir):
                   
          self.Tiles_Dir= Tiles_Dir
          self.folds = folds
          self.state = state       
          self.Project_Dir = Project_Dir
               
     def Generate(self):
          
          global DFs
          
          # For the training sessions******************************************
          if self.state == 'train':
               
               # For 3-fold data-----------------------------------------------
               if self.folds== 3:
               
                    DFs = []
                    
                    for file1 in glob.glob('*.xlsx'): 
                         
                         if (os.path.basename(self.Tiles_Dir) in file1):
                              
                              if 'fold' in file1 and str(5) not in file1:
                                                           
                                   DFs.append(file1)
                              
                    Fold_1 = pd.read_excel(DFs[0], index_col=0)
                    Fold_2 = pd.read_excel(DFs[1], index_col=0)
                    Fold_3 = pd.read_excel(DFs[2], index_col=0)
                    
                    Folds = [Fold_1, Fold_2, Fold_3]          
                              
                    return Folds;
               
               # For 5-fold data-----------------------------------------------
               elif self.folds==5:
                    
                    DFs = []
                    
                    for file1 in glob.glob('*.xlsx'): 
                         
                         if (os.path.basename(self.Tiles_Dir) in file1):
                              
                              if 'fold' in file1 and str(5) in file1:
                                                           
                                   DFs.append(file1)
                              
                    Fold_1 = pd.read_excel(DFs[0], index_col=0)
                    Fold_2 = pd.read_excel(DFs[1], index_col=0)
                    Fold_3 = pd.read_excel(DFs[2], index_col=0)
                    Fold_4 = pd.read_excel(DFs[3], index_col=0)
                    Fold_5 = pd.read_excel(DFs[4], index_col=0)
                    
                    Folds = [Fold_1, Fold_2, Fold_3, Fold_4, Fold_5]          
                              
                    return Folds;
               
               # For the whole training data
               elif self.folds=='None':
                    
                    DFs = []
                    
                    for file1 in glob.glob('*.xlsx'): 
                         
                         if (os.path.basename(self.Tiles_Dir) in file1):
                              
                              if 'TrainData' in file1:
                                                           
                                   DFs.append(file1)
                    
                    TrainData = pd.read_excel(DFs[0], index_col=0)
                    
                    return TrainData;
                                   
          # For the test sessions *********************************************          
          elif self.state == 'test':
                          
               DFs = []
               
               for file in glob.glob(self.Project_Dir + '\\' + os.path.basename(self.Tiles_Dir) + '_TestData.xlsx'): 
                    
                    if 'Test' in os.path.basename(file):
                         
                         DFs.append(file)
                         
               TestData = pd.read_excel(DFs[0], index_col=0)
               
               return TestData;
     