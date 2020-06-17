# Deep_Learning_Ovarian_Cancer_Image_Classification

Project: Histological and Molecular Classification of Ovarian Cancer H&E and Frozen Tumor Tissue Images

        Purpose: Steps and description to work with the scripts
        Author : Georgia Liapi (Master Student, Systems Biology, Maastricht University)
        Academic Year: 2019-2020
        Image Datasets: Ovarian Carcinomas Histopathology Dataset (Simon Fraser University) (SFU) from Köbel et al., 2010
                        TCGA-OV-DX (H&E-stained images)                                     (DX)
                        TCGA-OV-KR (Frosen Tumor Tissue images)                             (KR)
    
    Below are the steps and order of using each Script for this project.
    
    To begin with, 6 folders must be ready (2 per dataset): 
    
                    i) SFU Ovarian Cancer svs images --> ii) SFU Ovarian Cancer jpg images
                          iii) TCGA-OV DX svs images --> iv) TCGA-OV DX jpg images
                            v) TCGA-OV KR svs images --> vi) TCGA-OV KR jpg images
    
    >>> To obtain jpg images from the svs images:
        QuPath** was used for annotating the tumor in each svs image and tiling it into many smaller jpg images.  
        After the tiling process, for each of the three svs image datasets, a new directory is created with as 
        many subfolders as the number of the primary svs images. Each folder inside a new jpg image directory 
        corresponds to a patient and holds inside all the jpg images of the current patient's tumor.
        
        **Bankhead P, Loughrey MB, Fernández JA, Dombrowski Y, McArt DG, Dunne PD, et al. QuPath: Open source software for digital                 pathology image analysis. Sci Rep. 2017 Dec;7(1):16878
        
    Second, a directory must be structured: 
    
      a) Create a project folder and give a name (eg OV_PROJECT)
      b) Create 2 subfolders (RESULTS, SCRIPTS)
      c) Create 3 subfolders inside the 'RESULTS' folder 
         (HISTOLOGICAL_SUBTYPES, TCGA_DX_MOLECULAR_SUBTYPES, TCGA_KR_MOLECULAR_SUBTYPES)                                         
      d) move the provided project scripts to the SCRIPTS folder                                        
      e) move the table 'transcanadian_training and test set slides.xls' (SFU dataset) and                                          
         the 'NIHMS958065-supplement-4.xlsx' (Berger et al., 2018) also to the SCRIPTS folder
                                                 
    Install: Anaconda for Windows 64 (https://www.anaconda.com/)
             Visual Studio 2017 (https://visualstudio.microsoft.com/vs/older-downloads/) 
             cudnn for Windows (https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#install-windows)
             CUDA 10.0 ToolKit (https://developer.nvidia.com/cuda-10.0-download-archive)                  
             Download and save the provided 'requirements.txt', inside the project folder                 
             Open the 'Anaconda Prompt (Anaconda 3)' 

    Navigate to the project folder with:  cd C:\...\OV_PROJECT                
                  Create an environment:  conda create -y --name <Name> python==3.7.5
             Enter the environment with:  conda activate <Name>      
                            Install pip:  conda install -c anaconda pip                                               
         Install the 'requirements.txt':  pip install -r requirements.txt
                      Open Spyder. Type:  spyder
              
------------------------------------------------------ S T A R T -----------------------------------------------------------------------           
Inside Spyder:  On the top right, 'Browse a working directory' icon, select the folder 'SCRIPTS':
                      
Steps:  
(1) Open 'TCGAOV_MolecularData.py' (just script)

    This function will read the data (slides *.svs) from each directory (dataset).
          
     Set the directories on the top of the script(ONLY NEEDED ONCE):    
                                                           
                             # Set the directories of the svs files
                             OV_DX_SVS = '...\\TCGA_OV_DXsvs'
                             OV_KR_SVS = '...\\TCGA_OV_KRsvs'

                             Save the changes (CTRL+S)
          
          -> Run File  (F5)
          
         Outputs: an excel file after filtering the 'NIHMS958065-supplement-4.xlsx' to keep only the Ovarian Cancer Data (OV)
                  an excel file with the number of slides (patients) per class for the TCGA-OV-DX Dataset
                  an excel file with the number of slides (patients) per class for the TCGA-OV-KR Dataset
                  an excel file with each of the slides (patients) matched to its class for the TCGA-OV-DX Dataset
                  an excel file with each of the slides (patients) matched to its class for the TCGA-OV-KR Dataset
                                               
(2) Open 'Read_Dirs.py'     (function)
         
    Set the directories on the top of the script (ONLY NEEDED ONCE):

         # Set the tiles (*.jpg) directories for all datasets
         OV_SFU_JPG = '...\\SimonFraserUniversityJPG'
         OV_DX_JPG  = '...\\TCGA_OV_DX_JPG'
         OV_KR_JPG  = '...\\TCGA_OV_KR_JPG'

         Save the changes (CTRL+S)

         This function is utilized by the next one, 'Match_Patients_Classes.py'

         Output per run: an excel filewith 2 columns (TILE_NAME, FULL_PATH)    (a tile is an image)   

(3)   Open 'Match_Patients_Classes.py' (function)

     This is used to generate three excel files, each with three columns: 
     Tile_NAME, FULL_PATH, Class, and the slide info (patient info) as indices

     Set the directories on the top of the script (ONLY NEEDED ONCE):    

               # Set the tiles (*.jpg) directories for all datasets
               OV_SFU_JPG = '...\\SimonFraserUniversityJPG'
               OV_DX_JPG  = '...\\TCGA_OV_DX_JPG'
               OV_KR_JPG  = '...\\TCGA_OV_KR_JPG'

               Save the changes (CTRL+S)
                                                       
         -> Run File  (F5)  (necessary before using the console)                
          
         Use the function 'Match_Patients_Classes' three times (3 Datasets), in the CONSOLE:
         Type:
             1)  Match_Patients_Classes(Dirs[0], Current_Dir) (and Enter)
             2)  Match_Patients_Classes(Dirs[1], Current_Dir) (and Enter)
             3)  Match_Patients_Classes(Dirs[2], Current_Dir) (and Enter )                                               

             Output per run: an excel with 3 columns (TILE_NAME, FULL_PATH, Class)  
                                                  
(4) Open 'Test_Data.py'  (function)

    This is used to separate the whole data per Dataset into Test and Train data
         
    Set the directories on the top of the script (ONLY NEEDED ONCE):    

               # Set the tiles (*.jpg) directories for all datasets
               OV_SFU_JPG = '...\\SimonFraserUniversityJPG'
               OV_DX_JPG  = '...\\TCGA_OV_DX_JPG'
               OV_KR_JPG  = '...\\TCGA_OV_KR_JPG'

               Save the changes (CTRL+S)       

     -> Run File  (F5) (necessary before using the console)

     Use the function 'Test_Data' three times (3 Datasets), in the CONSOLE:
     Type:
          1)  Test_Data(Dirs[0], Current_Dir) (and Enter)
          2)  Test_Data(Dirs[1], Current_Dir) (and Enter)
          3)  Test_Data(Dirs[2], Current_Dir) (and Enter)   

          Outputs per run (all of them per run): a TrainData excel with 3 columns (TILE_NAME, FULL_PATH, Class) 
                                                 a TestData  excel with 3 columns (TILE_NAME, FULL_PATH, Class) 
                                                 an excel file with tthe number of Patients per Class in the TrainData
                                                 an excel file with tthe number of Patients per Class in the TestData

(5) Open 'k_folds.py'    (function)    

    This function will generate k-folds (3 or 5) from the 
    corresponding  '<...>_TrainData.xlsx' file in the project folder.
         
    Set the directories on the top of the script (ONLY NEEDED ONCE):    
               # Set the tiles (*.jpg) directories for all datasets
               OV_SFU_JPG = '...\\SimonFraserUniversityJPG'
               OV_DX_JPG  = '...\\TCGA_OV_DX_JPG'
               OV_KR_JPG  = '...\\TCGA_OV_KR_JPG'

               Save the changes (CTRL+S)       

     -> Run File  (F5) (necessary before using the console)

     Use the function 'k_folds' six times (2 * (3 Datasets)), in the CONSOLE:
     Type:
              1)  k_folds(Dirs[0], Current_Dir, folds=3) (and Enter)
              2)  k_folds(Dirs[1], Current_Dir, folds=3) (and Enter)
              3)  k_folds(Dirs[2], Current_Dir, folds=3) (and Enter)   
              4)  k_folds(Dirs[0], Current_Dir, folds=5) (and Enter)
              5)  k_folds(Dirs[1], Current_Dir, folds=5) (and Enter)
              6)  k_folds(Dirs[2], Current_Dir, folds=5) (and Enter)

              Output per run: with 'folds = 3' : 3 excel tables (3 folds) of data for the current Dataset
                          or  with 'folds = 5' : 5 excel tables (5 folds) of data for the current Dataset

(6) Open 'DataLoad.py'    (function)  

    This function will import the data used in the 'Train_Val_Pred.py' by using
    the data excel tables (folds) in the project folder.
         
    Set the directories on the top of the script (ONLY NEEDED ONCE):    

                # Set the tiles (*.jpg) directories for all datasets 
                OV_SFU_JPG = '...\\SimonFraserUniversityJPG'
                OV_DX_JPG  = '...\\TCGA_OV_DX_JPG'
                OV_KR_JPG  = '...\\TCGA_OV_KR_JPG'    

      Output per run: if state == 'train': with 'folds = 3' : returns 1 list of 3 dataframes for the selected Dataset
                                      or   with 'folds = 5' : returns1 list of 5 dataframes for the selected Dataset
                      if state == 'test' : returns 1 dataframe with the test data for the the selected Dataset

(7) 'Set_Network.py'    (function)        

     This function will be used in the 'Train_Val_Pred.py' (function)
     The arguments for 'Set_Network.py' are being set automatically inside 'Train_Val_Pred.py' 
     
(8) 'plot_metrics.py'  (function) 

    This function will be used in the 'Train_Val_Pred.py' (function)
    The arguments for 'plot_metrics.py' are being set automatically inside 'Train_Val_Pred.py'   
    
    || DISCLAIMER: THE FOLLOWING FUNCTION'S CONTENT WAS NOT A PRODUCT OF THE WORK OF GEORGIA LIAPI! || 
    || The following function's content is a plotting paradigm belonging to the 'tensorflow.org',                                     https://www.tensorflow.org/tutorials/images/classification?authuser=0&hl=zh-cn

(9) 'plot_roc.py'  (function) 

     This function will be used in the 'Train_Val_Pred.py' (function)
     The arguments for 'plot_roc.py' are being set automatically inside 'Train_Val_Pred.py'
     
    || DISCLAIMER: THIS FUNCTION WAS NOT A PRODUCT OF THE WORK OF GEORGIA LIAPI! ||     
    ** DISCLAIMER of Copyrights:
       The following function is the original plot_roc_curve() from the 'plotters.py' (lines 186-330) that belongs to: 
       reiinakano/scikit-plot repository (https://github.com/reiinakano/scikit-plot) **
    ||-------------------------------------------------------------------------------------------     
    || Contributors: Reiichiro Nakano, Christopher Wells, wikke;, lugq,  Ryan Joshua Liwag, Matthew Emery, Lj Miranda, Joshua Engelman,
                     Frank Herfert, Emre Can, Doug Friedman, Christopher Ren
                     
    Note: The adjusted plot_roc() is utilized for the current project, with some added arguments (Data, Idx) and minor changes in thE current lines 61-62.                 
                 
(10) 'calc_roc_auc.py'  (function) 

     This function will be used in the 'Train_Val_Pred.py' (function)
     The arguments for 'calc_roc_auc.py' are being set automatically inside 'Train_Val_Pred.py'               
     
     || DISCLAIMER: THIS FUNCTION WAS NOT A PRODUCT OF THE WORK OF GEORGIA LIAPI! || 
     ** DISCLAIMER of Copyrights:
        The following function is part of the original plot_roc_curve() from the 'plotters.py' (lines 186-330) that belongs to: 
        reiinakano/scikit-plot repository (https://github.com/reiinakano/scikit-plot) **
        
    Note: The following lines 26-46 come from the lines 243 to 260 of the original plot_roc_curve() of the reiinakano/scikit-ploT repository. 
     
(11) 'precision_recall.py'  (function) 

     This function will be used in the 'Train_Val_Pred.py' (function)
     The arguments for 'precision_recall.py' are being set automatically inside 'Train_Val_Pred.py' 
     
    || DISCLAIMER: THIS FUNCTION WAS NOT A PRODUCT OF THE WORK OF GEORGIA LIAPI! ||     
    ** DISCLAIMER of Copyrights:
       The following function is the original plot_precision_recall_curve() from the 'plotters.py' (lines 422-541) that belongs to: 
       reiinakano/scikit-plot repository (https://github.com/reiinakano/scikit-plot) **
    ||-------------------------------------------------------------------------------------------  
    || Contributors: Reiichiro Nakano, Christopher Wells, wikke;, lugq,  Ryan Joshua Liwag, Matthew Emery, Lj Miranda, Joshua Engelman,
                     Frank Herfert, Emre Can, Doug Friedman, Christopher Ren
    
       Note: The adjusted precision_recall() is utilized for the current project, with some added arguments (Data, Idx) and minor changes in the current lines 74-75.                
          
(12) 'Predict.py' (function)

     This function will be used in the 'Train_Val_Pred.py' or in the 'Train_whole_PredTest.py' 
     The arguments for 'Predict.py' are being set automatically inside 'Train_Val_Pred.py' and 'Train_whole_PredTest.py' 

     Outputs are automatically sent to the current dataset's Results' folder. 

     After a whole k-fold cross validation in the 'Train_Val_Pred.py', the output tables are: 
               
                          TABLE NAME                                                         COLUMNS CONTENT
      k  * 'experiment_network_Report_datatype_cvtype_fold<>.xlsx'                 (classification report, sklearn library) 
      k  * 'experiment_network_CM_Tiles_datatype_cvtype_fold<>.xlsx'               (confusion matrix on the images level, sklearn library)   
      k  * 'experiment_network_Predicted_datatype_cvtype_fold<>.xlsx'              (Predicted probabilities over classes per image, predicted label, True class)
      k  * 'experiment_network_Classification_Metrics_datatype_cvtype_fold<>.xlsx' (Number of correctly and falsely predicted classes per patient,
                                                                                    Classification accuracy and average predicted probability per patient)
      k  * 'experiment_network_NumOfClassPerID_datatype_cvtype_fold<>.xlsx'        (number and name of predicted classes per patient, true class per patient)    
      k  * 'experiment_network_CM_Pnt_datatype_cvtype_fold<>.xlsx'                 (confusion matrix on the patient level, number of correct and
                                                                                    false classified patients per class)                                                                    
(13) Open 'Train_Val_Pred.py'  (function) 
    
     Set the directories on the top of the script (ONLY NEEDED ONCE):

                  # Set the tiles (*.jpg) directories          
     Here -->     OV_SFU_JPG = '...\\SimonFraserUniversityJPG'
     Here -->     OV_DX_JPG  = '...\\TCGA_OV_DX_JPG'
     Here -->     OV_KR_JPG  = '...\\TCGA_OV_KR_JPG'

                  Dirs = [OV_SFU_JPG, OV_DX_JPG, OV_KR_JPG] 

                  # Set the results directories
     Here -->     OV_SFU = '...\\OV_PROJECT\\RESULTS\\HISTOLOGICAL_SUBTYPES'
     Here -->     OV_DX  = '...\\OV_PROJECT\\RESULTS\\TCGA_DX_MOLECULAR_SUBTYPES'
     Here -->     OV_KR  = '...\\OV_PROJECT\\RESULTS\\TCGA_DX_MOLECULAR_SUBTYPES'

                  Results_Dirs  = [OV_SFU, OV_DX, OV_KR] 
                                                                                       
     ---> Run File  (F5) (necessary before using the console)

     >> A R G U M E N T S    F O R   Train_Val_Pred():

          1) the image tiles dorectory (Tiles_Dir = Dirs[0] or Tiles_Dir = Dirs[1] or Tiles_Dir = Dirs[2]) -->the whole expression, '=' is needed
          2) the number of folds       (3 or 5, just as integer)
          3) the network               ('ResNet50' or 'InceptionV3' or 'Xception')
          4) the results' directories  (Results_Dirs = Results_Dirs[0] or Results_Dirs = Results_Dirs[1] or Results_Dirs = Results_Dirs[2]) --> the whole expression, with '=' is needed
          5) Project_Dir= Current_Dir  needs no action
          6) experiment                ('Hist' or 'Mol')  
          7) cvtype                    ('_3CV_Sess_' or '_5CV_Sess_')
          8) DataType                  ('last_fold')

     >> Finally, to Run 'Train_Val_Pred.py', e.g. for the OC dataset, 3-fold cross validation and Xception, type in the CONSOLE:

        Train_Val_Pred(Tiles_Dir=Dirs[0], folds=3, network='Xception', Results_Dirs=Results_Dirs[0], Project_Dir= Current_Dir, experiment='Hist', cvtype='_3CV_Sess_', DataType='last_fold') 
                                                                     
                                                         (and hit Enter)

    ==> R E P E A T    F O R    E A C H    D A T A S E T,    N E T W O R K   A N D    K - F  O L D    T Y P E         

(14) Open 'Train_whole_Test.py'  (function) 

     Set the directories on the top of the script (ONLY NEEDED ONCE):

                  # Set the tiles (*.jpg) directories          
     Here -->     OV_SFU_JPG = '...\\SimonFraserUniversityJPG'
     Here -->     OV_DX_JPG  = '...\\TCGA_OV_DX_JPG'
     Here -->     OV_KR_JPG  = '...\\TCGA_OV_KR_JPG'

                  Dirs = [OV_SFU_JPG, OV_DX_JPG, OV_KR_JPG] 

                  # Set the results directories
     Here -->     OV_SFU = '...\\OV_PROJECT\\RESULTS\\HISTOLOGICAL_SUBTYPES'
     Here -->     OV_DX  = '...\\OV_PROJECT\\RESULTS\\TCGA_DX_MOLECULAR_SUBTYPES'
     Here -->     OV_KR  = '...\\OV_PROJECT\\RESULTS\\TCGA_DX_MOLECULAR_SUBTYPES'

                  Results_Dirs  = [OV_SFU, OV_DX, OV_KR] 
                                                                                       
    ---> Run File  (F5) (necessary before using the console)
         
    >> A R G U M E N T S    F O R   Train_whole_Test():

          1) the image tiles dorectory (Tiles_Dir = Dirs[0] or Tiles_Dir = Dirs[1] or Tiles_Dir = Dirs[2]) -->the whole expression, '=' is needed
          2) the network               ('ResNet50' or 'InceptionV3' or 'Xception')
          3) the results' directories  (Results_Dirs = Results_Dirs[0] or Results_Dirs = Results_Dirs[1] or Results_Dirs = Results_Dirs[2]) --> the whole expression, with '=' is needed
          4) Project_Dir= Current_Dir  needs no action
          5) experiment                ('Hist' or 'Mol')  
          6) cvtype                    ('_FinalTest_')
          7) DataType                  ('TestData')       

     >> Finally, to Run 'Train_whole_Test.py', e.g. for the OC dataset and Xception, type in the CONSOLE:

     Train_whole_Test(Tiles_Dir=Dirs[0], network='Xception', Results_Dirs=Results_Dirs[0], Project_Dir= Current_Dir, experiment='Hist', cvtype='_FinalTest_', DataType='TestData') 
                                                          
                                                          (and hit Enter)
-----------------------------------------------------------------   E N D ------------------------------------------------------------------
