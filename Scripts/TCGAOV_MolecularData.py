"""
        Author: Georgia Liapi, Master Student in Systems Biology, Maastricht University
 Academic Year: 2019-2020
       Project: 'Deep learning-based prediction of molecular subtypes in human ovarian cancer'
      Datasets: TCGA-OV-DX
                TCGA-OV-KR
       Utility: This script imports the OV molecular annotation (OV_Subtypes) from the Berger et al. (2018) study, finds the patients in the TCGA-OV-DX and -KR slides,  for which
                there are matched labels and finds the number of patients (slides) per Label for both Datasets
"""

# Import the required libraries
import pandas as pd
import glob, os
from difflib import SequenceMatcher
from collections import Counter 
import xlrd

# Set working directory to the tables folder
Current_Dir = os.getcwd()

# Set the directories of the *.svs whole slide images
# The 'NIHMS958065-supplement-4.xlsx' table must be in the 'SCRIPTS' folder
OV_DX_SVS = '..\\TCGA_OV_DXsvs'
OV_KR_SVS = '..\\TCGA_OV_KRsvs'

#------------------------------------------------------------------------------
#                         Molecular Data Excel
#------------------------------------------------------------------------------

# Import the Berger molecular annotation
Mol_Data_excel = pd.read_excel(Current_Dir + '\\' + 'NIHMS958065-supplement-4.xlsx', index=0, header=1)

# Filter to keep only the OV data
OV_excel   = Mol_Data_excel.loc[Mol_Data_excel['Tumor.Type'] == 'OV']  
OV_excel   = OV_excel.reset_index(drop=True)
OV_excel["Sample.ID"].head()
len(OV_excel)

# Save the table matching Slides to their labels in the 'OV_Subtype' column
OV_excel.to_excel(Current_Dir + '\\' + "OV_Slides_To_Labels_NoEdit.xlsx")


#------------------------------------------------------------------------------
#                 TCGA-OV DX Slides with molecular annotation
#------------------------------------------------------------------------------
# Import slides' paths from one great directory
Path = []
for lb in glob.glob(OV_DX_SVS + '\\' +'*.svs'): 
    Path.append(lb)

#Check content    
len(Path)
Path[0]

# Extract names from the slides in TCGA_OV_DXsvs
BaseName = []
for i in Path[0:len(Path)]:
   baseName = os.path.basename(i)
   BaseName.append(baseName)

# Turn the BaseNames list into a dataframe
# So far the BaseName contains all patients (slides) downloaded by the TCGA-OV-DX (both having a matched Molecular label and not)
BaseName = pd.DataFrame(BaseName)
BaseName.columns = ["SlideName"]


#-----------------------------------------------------------------------------------------------------------------------
#  Find for which of the H&E slides in TCGA-OV-DX there are available molecular data in the OV_excel ('NIHMS958065-supplement-4.xlsx')
# ----------------------------------------------------------------------------------------------------------------------

# Insert new column to BaseName
BaseName["Molecular_Subtypes"] = ""

# Make a loop to find slides existing in the table
for sld,sldobj in enumerate(OV_excel["Sample.ID"]):
     
     # The longest common string part between the slide names in the OV_excel table and those in the TCGA-OV-DX *.svs directory
     for SLD, SLDobj in enumerate(BaseName["SlideName"]):
        match = SequenceMatcher(None, sldobj, SLDobj).find_longest_match(0, 
                               len(sldobj), 0, len(SLDobj))
        
        # For any common slide name in both the OV_excel table and the TCGA-OV-DX *.svs directory, 
        # put the current molecular subtype of the row of the OV_excel table, on a new column (current, "Molecular_Subtypes") and name row in the *.svs directory table (BaseName)
        if sldobj == OV_excel["Sample.ID"][sld][match.a: match.a + match.size]:
             
             # The 13nth column is the column of interest in the 'NIHMS958065-supplement-4.xlsx' table (molecular annotation)
             BaseName["Molecular_Subtypes"][SLD] = OV_excel.iloc[sld, 13]
             print("Match Found!")
        else:
              print("Continue Searching:     Table IDs with Class: {},  Slides: {} ".format(str(sld), str(SLD)))       

print(BaseName.loc[[0,1,2,3]])      

# ----------------------------------------------------------------------------------------------
#  Filter the BaseName second column to keep only slides that are matched to a molecular subtype
# ----------------------------------------------------------------------------------------------

# Instantiate an empty dataframe 
Refined = pd.DataFrame(data= {"Slide" : [] , "Mol_Subtypes" : []})

# Save in Types the four unique ovarian cancer molecular subtypes
Types = list(BaseName["Molecular_Subtypes"].unique())

# *** If there is an icon of typing error on the left, ignore it. It is working! ***
Types.remove(nan)
Types.remove(str(''))

# Run through the "Molecular_Subtypes" column, and keep only the rows where there is a class (molecular subtype)
# Any row with empty molecular subtype entry will be filtered out
for IDX, SbTp in enumerate(BaseName["Molecular_Subtypes"]):
     if SbTp in Types:
       Refined.loc[IDX] = BaseName["SlideName"][IDX], SbTp

# Save the table with where each patient (slide) is matched to a label in the 'OV_Subtype' column
Refined.to_excel(Current_Dir + '\\' + "OVDXSlidesMolSub_Filtered.xlsx")

#------------------------------------------------------------------------------     
#                      Slides Per Class DX (subtypes)  
#------------------------------------------------------------------------------       
# Count how many different names exist within each of the ovarian cancer molecular subtypes in
#  the 'Mol_Subtypes' column of the Refined table
Molecular_Classes = Counter(Refined['Mol_Subtypes'])

# Make a dataframe
Molecular_Classes = pd.DataFrame(Molecular_Classes, index=[0])

# Save a table with the number of slides per Label
Molecular_Classes.to_excel(Current_Dir + '\\' + "OVDX_SlidesPerClass.xlsx")


#------------------------------------------------------------------------------
#                 TCGA-OV KR Slides with molecular annotation
#------------------------------------------------------------------------------
# Import slides' paths from one great directory
Path1 = []
for lb1 in glob.glob(OV_KR_SVS + '\\' + '*.svs'): 
    Path1.append(lb1)
len(Path1)
Path1[0]

# Extract the slides names 
BaseName1 = []
for i in Path1[0:len(Path1)]:
   baseName = os.path.basename(i)
   BaseName1.append(baseName)

BaseName1 = pd.DataFrame(BaseName1)
BaseName1.columns = ["SlideName"]


# -----------------------------------------------------------------------------------------
# Find for which of the Frozen Tissue slides in TCGA-OV there are available molecular data
# -----------------------------------------------------------------------------------------    
# Insert new column to BaseName1
BaseName1["Molecular_Subtypes"] = ""


#  Make a loop to find slides existing in the table
#  The following loop works as the one in lines 75-90
for sld1,sldobj1 in enumerate(OV_excel["Sample.ID"]):
     
     for SLD1, SLDobj1 in enumerate(BaseName1["SlideName"]):
        match = SequenceMatcher(None, sldobj1, SLDobj1).find_longest_match(0, 
                               len(sldobj1), 0, len(SLDobj1))
        
        if sldobj1 == OV_excel["Sample.ID"][sld1][match.a: match.a + match.size]:

             BaseName1["Molecular_Subtypes"][SLD1] = OV_excel.iloc[sld1, 13]
             print("Match Found!")
             
        else:
             print("Continue Searching:      Table IDs with Class: {},  Slides: {} ".format(str(sld1), str(SLD1)))    
             

# Filter the BaseName1 second column to keep only slides that are matched to a molecular subtype
Refined_KR = pd.DataFrame(data= {"Slide" : [] , "Mol_Subtypes" : []})
Types = list(BaseName1["Molecular_Subtypes"].unique())

# If there is an icon of typing error on the left, ignore it. It is working!
Types.remove(nan)
Types.remove(str(''))

for IDX1, SbTp1 in enumerate(BaseName1["Molecular_Subtypes"]):
     if SbTp1 in Types:
       Refined_KR.loc[IDX1] = BaseName1["SlideName"][IDX1], SbTp1

# Save the table with where each patient (slide) is matched to a label in the 'OV_Subtype' column
Refined_KR.to_excel(Current_Dir + '\\' + 'OVKRSlidesMolSub_Filtered.xlsx')
     

#------------------------------------------------------------------------------     
#                      Slides Per Class KR (subtypes)  
#------------------------------------------------------------------------------           
Molecular_Classes = Counter(Refined_KR['Mol_Subtypes'])
Molecular_Classes = pd.DataFrame(Molecular_Classes, index=[0])

# Save a table with the number of slides per Label
Molecular_Classes.to_excel(Current_Dir + '\\' + "OVKR_SlidesPerClass.xlsx")

   
