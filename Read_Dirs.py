# -*- coding: utf-8 -*-
"""
         Author: Georgia Liapi, Master Student in Systems Biology, Maastricht University
  Academic Year: 2019-2020
        Project: 'Deep learning-based prediction of molecular subtypes in human ovarian cancer'
        Utility: Load and Aggregate Tile Paths and Names in an excel file for each jpg Image Tiles Dataset (one dataset per run)
        
           Note: a txt file, with TCGA-OV-KR slides that need to be kept out, needs to be placed in the Project Folder (OV_PROJECT\\SCRIPTS\\KR_SlidesBlur.txt)
"""

# Import Required Libraries
import pandas as pd
import glob2,glob
import os

# Get the working directory
Current_Dir = os.getcwd()

# Set the tiles (*.jpg) directories for all datasets
OV_SFU_JPG = '..\\SimonFraserUniversityJPG'
OV_DX_JPG  = '..\\TCGA_OV_DX_JPG'
OV_KR_JPG  = '..\\TCGA_OV_KR_JPG'

# Make a list of all the datasets jpg image directories
Dirs = [OV_SFU_JPG, OV_DX_JPG, OV_KR_JPG] 


# Define a function to read the image tiles in a folder and match them to the patients and classes (ovarian cancer subtypes)
def Read_Dirs(Tiles_Dirs=Dirs, Project_Dir = Current_Dir):

     # These variables can be reachable outside the function, in case we want to check their content
     global Tiles_Path, BaseNames,ClassPerID, DataFrame 
     
     # For the Ovarian Cancer Histological Subtypes Image Tiles Dataset
     if Tiles_Dirs == Dirs[0]:
          # Import tiles from one great directory-------------------------------
          Tiles_Path = []
          for lbl in glob2.glob(Tiles_Dirs + '\\**\\*.jpg'): 
              Tiles_Path.append(lbl)
          print('Number of Tiles: {} in the {} Folder'.format(len(Tiles_Path), os.path.basename(Tiles_Dirs)))
          
          # Fix the names replace u with U (THERE WERE TYPING ERRORS IN THE PRIMARY TABLE)
          for kk in range(0, len(Tiles_Path)):
               Tiles_Path[kk] = str(Tiles_Path[kk]).replace("u", "U", 1)
          
          # Extract the image tiles basenames--------------------------------------------------
          BaseNames = []
          for i in Tiles_Path [0:len(Tiles_Path)]:
             baseNames = os.path.basename(i)
             BaseNames.append(baseNames)
             
          # Create a dataframe with the Tiles' Names and their tiles' full paths
          DataFrame = pd.DataFrame(data = {"TILE_NAME": BaseNames, "FULL_PATH": Tiles_Path})
          DataFrame.to_excel(r'{}\\{}.xlsx'.format(Project_Dir,os.path.basename(Tiles_Dirs)), index=False)
     
     # For the TCGA-OV-DX Image Tiles Dataset     
     elif Tiles_Dirs == Dirs[1]:
          # Import tiles from one great directory-------------------------------
          Tiles_Path = []
          for lbl in glob2.glob(Tiles_Dirs + '\\**\\*.jpg'): 
              Tiles_Path.append(lbl)
          print('Number of Tiles: {} in the {} Folder'.format(len(Tiles_Path), os.path.basename(Tiles_Dirs)))          
          
          # Extract the tiles basenames--------------------------------------------------
          BaseNames = []
          for i in Tiles_Path [0:len(Tiles_Path)]:
             baseNames = os.path.basename(i)
             BaseNames.append(baseNames)
             
          # Create a dataframe with the Tiles' Names and their tiles' full paths
          DataFrame = pd.DataFrame(data = {"TILE_NAME": BaseNames, "FULL_PATH": Tiles_Path})
          DataFrame.to_excel(r'{}\\{}.xlsx'.format(Project_Dir,os.path.basename(Tiles_Dirs)), index=False)
     
     # For the TCGA-OV-KR Image Tiles Dataset          
     elif Tiles_Dirs == Dirs[2]:
          
          # Import the folder names for the KR images
          subfolders = [folder.name for folder in os.scandir(Tiles_Dirs) if folder.is_dir()]
          
          #The following txt file contains slides that have to be removed due to being blurred (found during tumor annotation in Qupath)
          Blur_Slides = pd.read_csv(r'C:\Users\g_lia\Desktop\OV_PROJECT\SCRIPTS\KR_SlidesBlur.txt', header = None)
          
          # Remove the blur images 
          Common= list(set(subfolders).intersection(Blur_Slides[0][:]))
          
          Subfolders = pd.DataFrame(subfolders, index = subfolders)
          Subfolders = Subfolders.drop(index= Common) 
          Subfolders = Subfolders.reset_index(drop=True)
          
          # Import tiles from one great directory-------------------------------
          Tiles_Path = []
          for folder in Subfolders.iterrows():             
               for lbl in glob.glob(Tiles_Dirs + '\\' + folder[1][0] + '\\*.jpg'): 
                   Tiles_Path.append(lbl)
          print('Number of Tiles: {} in the {} Folder'.format(len(Tiles_Path), os.path.basename(Tiles_Dirs)))          
            
          # Extract the tiles basenames--------------------------------------------------
          BaseNames = []
          for i in Tiles_Path [0:len(Tiles_Path)]:
             baseNames = os.path.basename(i)
             BaseNames.append(baseNames)
                                     
          # Create a dataframe with the Tiles' Names and their tiles' full paths
          DataFrame = pd.DataFrame(data = {"TILE_NAME": BaseNames, "FULL_PATH": Tiles_Path})                                                             
          DataFrame.to_excel(r'{}\\{}.xlsx'.format(Project_Dir,os.path.basename(Tiles_Dirs)), index=False)
          
     return DataFrame;
 
   
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     