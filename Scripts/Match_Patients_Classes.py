# -*- coding: utf-8 -*-
"""
        Author: Georgia Liapi, Master Student in Systems Biology, Maastricht University
 Academic Year: 2019-2020
       Project: 'Deep learning-based prediction of molecular subtypes in human ovarian cancer'
       Utility: Load the tiles excel, match the patients(tiles) to their classes, extract a new table with a new column of classes (Ovarian Cancer Subtypes)
"""

# Import Required Libraries
import pandas as pd
from difflib import SequenceMatcher
import os, openpyxl
from  Read_Dirs import Read_Dirs

"""
Define a function that takes as inputs the jpg Image Tiles dataframe from Read_Dirs() and the Provided Class Data and 
returns a new Dataframe with one more column that matches Patients' Image Tiles to their classes (One-the same for all images of a patient).
The SIMON FRASER UNIVERSITY OVARIAN CANCER H&E histological dataset (Köbel et al., 2010) are imported here; the TCGA-OV Molecular data were incorporated in TCGAOV_MolecularData().
"""     

# Get the working directory
Current_Dir = os.getcwd()

# Set the tiles (*.jpg) directories for all datasets
OV_SFU_JPG = '..\\SimonFraserUniversityJPG'
OV_DX_JPG  = '..\\TCGA_OV_DX_JPG'
OV_KR_JPG  = '..\\TCGA_OV_KR_JPG'
Dirs = [OV_SFU_JPG, OV_DX_JPG, OV_KR_JPG] 


def Match_Patients_Classes(Tiles_Dir= Dirs, Project_Dir= Current_Dir):
     
     global Tiles_DF, IDstoClasses, DataFrame, DataFrame_New
     
     
     # -------------------------------
     # For the Simon Fraser University OV Dataset (Köbel et al., 2010)
     # -------------------------------
     if Tiles_Dir== Dirs[0]:
          
          # Load the primary data (image fullpaths and basenames) from running the function Read_Dirs()
          DataFrame = Read_Dirs(Dirs[0],  Current_Dir)
          
          #Import the Class Data
          ClassData = pd.read_excel(Current_Dir + '\\' + 'transcanadian_training and test set slides.xls', index=0)
          
          # Extract the simplest part of the patient IDs
          
          #Import the tiles dataframe     
          Tiles_DF = DataFrame
               
          # Refine Patient IDs (YYY + #...# --> YYY to YYY###)
          IDs1 = ClassData["Unnamed: 2"]+  ClassData["ID"].map(str)
          IDs1toClass = pd.DataFrame(data = {"ID": IDs1, "Class":  ClassData["Diagnosis"]}, index=None)
          
          IDs2 = ClassData["Unnamed: 7"]+ ClassData["ID.1"].map(str)
          IDs2toClass = pd.DataFrame(data = {"ID": IDs2, "Class": ClassData["Diagnosis.1"]}, index=None)
          
          IDstoClasses = IDs1toClass.append(IDs2toClass)
          IDstoClasses = IDstoClasses.reset_index(drop=True)
     
          # Added value : generic way with minor adjustments
          Tiles_DF["Class"] = ""
          
          # Find the longest common string part between the names in the Ovarian cancer study (Köbel et al., 2010) given Table and 
          # the image tile names in the histological subtypes jpg image directory 
          # ESSENTIALLY, FIND THE COMMON NAMES
          for idx1, part1 in enumerate(IDstoClasses["ID"]):
               for idx2, part2 in enumerate(Tiles_DF["TILE_NAME"]):
                     
                  common = SequenceMatcher(None, part1, part2).find_longest_match(0, len(part1),
                                          0, len(part2))
                  
                  # More Simply: if part1 equals the result from the comparison of part1 and part2
                  if part1 == IDstoClasses["ID"][idx1][common.a: common.a + common.size]:
                       
                       print("Match Found!")
                       # Put the corresponding label primarily found in the IDstoClasses table, to the Tiles_DF["Class"] correspoding patient row
                       Tiles_DF["Class"][idx2] = IDstoClasses.iloc[idx1, 1]
                       
                  else:
                       print("Continue Searching {} of the {} IDs, {} of the {} Tiles Names".format(idx1, len(IDstoClasses.index.unique()), idx2, len(Tiles_DF)))
     
     
          # Fix the indices to the patient 'ids' for an easier later indexing-------
          indices = []
          for idx, item in enumerate(Tiles_DF["TILE_NAME"]):
               if 'test' in str(Tiles_DF["TILE_NAME"][idx]).split('_', 1)[1]:          
                    indices.append(str(Tiles_DF["TILE_NAME"][idx]).split('st_', 2)[1])
               else:
                    indices.append(str(Tiles_DF["TILE_NAME"][idx]).split('_', 1)[1])   
                    
          indxs = []
          for IDX, ITEM in enumerate(indices):
               indxs.append(str(indices[IDX]).split('_', 1)[0])
               if 'svs' in indxs[IDX]:
                    indxs[IDX] = str(indxs[IDX].split('.', 1)[0])   
          Tiles_DF.set_index(pd.Series(indxs), inplace=True)
          
          # Save the csv file with the names, paths and the matched classes
          Tiles_DF.to_excel(r'{}\\{}.xlsx'.format(Project_Dir,os.path.basename(Tiles_Dir) + '_Matched'), index=True)
          
          return Tiles_DF;
      
     # --------------------------
     # For the TCGA-OV-DX Dataset
     # --------------------------
     elif Tiles_Dir== Dirs[1]:
          
          # Load the primary data (image fullpaths and basenames) from running the function Read_Dirs()
          DataFrame = Read_Dirs(Dirs[1],  Current_Dir)
          
          #Import the Class Data
          ClassData = pd.read_excel(Current_Dir + '\\' + 'OVDXSlidesMolSub_Filtered.xlsx', index_col=0)
          ClassData.drop(index=ClassData.index)
          ClassData = ClassData.reset_index(drop=True)
          
          # Create a dataframe with the simplest representative part of the patient IDs 
          IDstoClasses = pd.DataFrame(data={'Slide': '', 'Mol_Subtypes': ClassData['Mol_Subtypes'] })

          for row, name in enumerate(ClassData['Slide']):
               IDstoClasses['Slide'][row] = name.split('-',1)[1].split('.',1)[0].split('-',2)[0] + '-' + name.split('-',1)[1].split('.',1)[0].split('-',2)[1]           
          
          # Fix the indices to the patient 'ids' for an easier later indexing-------
          indices = []
          for idx, item in enumerate(DataFrame["TILE_NAME"]):
                 
              indices.append(item.split('-',1)[1].split('.',1)[0].split('-',2)[0] + '-' + item.split('-',1)[1].split('.',1)[0].split('-',2)[1])                          
          DataFrame.set_index(pd.Series(indices), inplace=True)
          
          # Filter the dataframe and keep only tiles with matched classes
          Indices = set(list(DataFrame.index.unique())) - set(list(IDstoClasses['Slide']))
          DataFrame = DataFrame.drop(index=list(Indices))
          
          # Match the molecular subtypes (classes) to the image tiles 
          DataFrame['Class'] = ''
          
          for idx1, part1 in enumerate(IDstoClasses['Slide'].unique()):
               
               DataFrame['Class'][part1] = IDstoClasses['Mol_Subtypes'][idx1]             
          
          # Save the csv file with the names, paths and the matched classes
          Tiles_DF =  DataFrame     
          Tiles_DF.to_excel(r'{}\\{}.xlsx'.format(Project_Dir,os.path.basename(Tiles_Dir) + '_Matched'), index=True)
           
          return Tiles_DF;
      
     # --------------------------
     # For the TCGA-OV-KR Dataset
     # --------------------------
     elif Tiles_Dir== Dirs[2]:
          
          # Load the primary data (image fullpaths and basenames) from running the function Read_Dirs()
          DataFrame = Read_Dirs(Dirs[2],  Current_Dir)
          
          #Import the tiles dataframe     
          Tiles_DF = DataFrame
          
          #Import the Class Data
          ClassData = pd.read_excel(Current_Dir + '\\' + 'OVKRSlidesMolSub_Filtered.xlsx', index_col=0)
          ClassData.drop(index=ClassData.index)
          ClassData = ClassData.reset_index(drop=True)
          
          # *** Create a dataframe with the simplest representative part of the patient IDs ***
          # Explanation about the anonymous identification understanding was found on https://docs.gdc.cancer.gov/Encyclopedia/pages/TCGA_Barcode/
          IDstoClasses = pd.DataFrame(data={'Slide': '', 'Mol_Subtypes': ClassData['Mol_Subtypes'] })

          for row, name in enumerate(ClassData['Slide']):
               IDstoClasses['Slide'][row] = name.split('-',1)[1].split('.',2)[0].split('-',2)[0] + '-' + name.split('-',1)[1].split('.',2)[0].split('-',2)[1]                 
          
          # Fix the indices to the patient 'ids' for an easier later indexing-------
          indices = []
          for idx, item in enumerate(Tiles_DF["TILE_NAME"]):
                 
              indices.append(item.split('-',1)[1].split('.',2)[0].split('-',2)[0] + '-' + item.split('-',1)[1].split('.',2)[0].split('-',2)[1])                          
          Tiles_DF.set_index(pd.Series(indices), inplace=True)
          
          # Filter the dataframe and keep only tiles with matched classes
          Indices = set(list(Tiles_DF.index.unique())) - set(list(IDstoClasses['Slide']))
          Tiles_DF = Tiles_DF.drop(index=list(Indices))
          
          # Match the molecular subtypes (classes) to the image tiles 
          Tiles_DF['Class'] = ''
          
          for idx1, part1 in enumerate(IDstoClasses['Slide'].unique()):
               
               Tiles_DF['Class'][part1] = IDstoClasses['Mol_Subtypes'][idx1]                             
             
          # Save the xlsx file with the names, paths and the matched classes
          Tiles_DF.to_excel(r'{}\\{}.xlsx'.format(Project_Dir,os.path.basename(Tiles_Dir) + '_Matched'), index=True)
                    
          return Tiles_DF;
               
               
               
               
               
               
     
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               