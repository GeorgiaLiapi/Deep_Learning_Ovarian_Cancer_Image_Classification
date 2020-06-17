# -*- coding: utf-8 -*-
"""
          Author: Georgia Liapi, Master Student in Systems Biology, Maastricht University
   Academic Year: 2019-2020
         Project: 'Deep learning-based prediction of molecular subtypes in human ovarian cancer.'
         Utility: Re-training 'Xception' with the Histological Subtypes Dataset (OC) and generate predictions on the Test Data (never used for training before by any of the CNNs)          

Available Usages: The current function can be used for any of the three datasets and neural networks in the current project
        Versions: CUDA 10.0, Keras: 2.2.4, Tensorflow-gpu: 1.15
"""

# Import the required libraries
from __future__ import print_function
from tensorflow import keras
from keras.layers import *
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras_preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import image
from keras.regularizers import l2
from keras.models import Sequential
from keras.utils import layer_utils
from keras import regularizers
from keras import backend as K
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle  
import sklearn
from collections import OrderedDict, Counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import random, h5py, os, openpyxl
import ipykernel
import PIL
import scikitplot 
 
# Import adjusted functions used for this project belonging 
# to the 'sklearn.metrics' library ! 
from plot_roc import  plot_roc
from precision_recall import plot_precision_recall_curve

# Personally created functions for this project
from Set_Network import Set_Network
from DataLoad import DataGen
from Predict import Predict 

#------------------------------------------------------------------------------
#                             S T A R T
#------------------------------------------------------------------------------
# Get the working directory
Current_Dir = os.getcwd()

# Set the tiles (*.jpg) directories for all datasets
OV_SFU_JPG = '..\\SimonFraserUniversityJPG'
OV_DX_JPG  = '..\\TCGA_OV_DX_JPG'
OV_KR_JPG  = '..\\TCGA_OV_KR_JPG'
Dirs = [OV_SFU_JPG, OV_DX_JPG, OV_KR_JPG] 

# Set the results directories
OV_SFU = '..\\Users\\g_lia\\Desktop\\OV_PROJECT\\RESULTS\\HISTOLOGICAL_SUBTYPES'
OV_DX  = '..\\Users\\g_lia\\Desktop\\OV_PROJECT\\RESULTS\\TCGA_DX_MOLECULAR_SUBTYPES'
OV_KR  = '..\\Users\\g_lia\\Desktop\\OV_PROJECT\\RESULTS\\TCGA_KR_MOLECULAR_SUBTYPES'
Results_Dirs  = [OV_SFU, OV_DX, OV_KR] 


#----------------------------------------------------------------------------------------------
# TRAIN A CNN WITH THE WHOLE TRAINING DATA AND PREDICT ON THE TESTING DATA (INFO IN THE README)
#                    (Used for any of the 3 Image datasets) 
#----------------------------------------------------------------------------------------------

def Train_whole_Test(Tiles_Dir, network, Results_Dirs, Project_Dir= Current_Dir, experiment='Hist', cvtype='_FinalTest_', DataType='TestData'):

     # Set parameters
     seed      = 42
     batchsize = 150
          
                       
          
     # Load the data - USE OF THE CREATED CLASS 'DataGen' ----------------
     # DataGen will generate one data fold for the Training and one data fold for the Testing
     Train_Data       = DataGen(Tiles_Dir, 'None', 'train', Project_Dir).Generate()

     Test_Data        = DataGen(Tiles_Dir, 'None', 'test', Project_Dir).Generate()
                                                 
     #-----------------------------------------------------------------------------------------------------------------------
     #                **********   Training on the whole training data of the selected dataset and Predicting on the primarily kept-out test data **********       
     #  This is a main loop that will run as many times as the number of the selected folds when calling the Train_Val_Pred()
     #-----------------------------------------------------------------------------------------------------------------------
 
     # Clear the previous session to control for OOM Errors in every next run
     K.clear_session()
               
     # Set the target size needed for the ImageDataGenerators
     if network == 'Xception' or network=='InceptionV3':               
           targetsize = (299,299)                                           
     else:               
           targetsize = (224,224)                
                                     
     # ------------------------------------------------------------------------------------------------------------------
     #                                             Re-sampling and shuffling data
     # ------------------------------------------------------------------------------------------------------------------

     # NOTE: The internal validation data can be used for final evaluation, as they are not used in the training to update the gradient descent
     # Set the validation and test data (THE SAME TABLE IS USED FROM BOTH)
     PredDF  = shuffle(shuffle(Test_Data))
            
     # ------------------------------------------------------------------------------------------------------------------
     #                                            Re-sampling the Train Data
     #                                         Same number of examples per Class
     # ------------------------------------------------------------------------------------------------------------------
     
     # Dictionary of class keys and their number of examples -imbalanced
     Imbalanced_Classes_TrainData = Counter(Train_Data['Class'])      
     print('\n The Training tiles per Class before re-sampling:{}'.format(Imbalanced_Classes_TrainData))
     
     # Find the number of examples in the minority class
     Minority_class_Train= np.array(list(Imbalanced_Classes_TrainData.values())).min()
     
     # Instantiate an new Train data table
     Train_resampled = pd.DataFrame()          
     
     # Select as many examples per Class as those in the minority class 
     # Essentially rows are selected from the primary Train data, but it cannot be controled that also each patient will contribute the same number of images
     for theClass in list(Imbalanced_Classes_TrainData.keys()):
          Train_resampled =  Train_resampled.append(Train_Data[Train_Data['Class']== theClass].sample(n=Minority_class_Train,replace=False)) 
          
     # Dictionary of class keys and their number of examples-balanced
     Balanced_Classes_Train = Counter(Train_resampled['Class'])      
     print('\n The Training tiles per Class after Under-sampling:{}'.format(Balanced_Classes_Train))
     
     # Finally, very important! ---> SHUFFLE THE TRAIN DATA 
     TrainDF = shuffle(shuffle(Train_resampled, random_state= 42))
                        
                       
     # ------------------------------------------------------------------------------------------------------------------                  
     #                                        Prepare the Image data generators
     #                                       The data will flow from dataframes !!!
     #          Dataframe Structure:  (column A: Images Full Paths, column B: Class as String) 
     #-------------------------------------------------------------------------------------------------------------------
     # T-R-A-I-N-I-N-G 
     print(" \n The training samples are: ")   
     train_gen = ImageDataGenerator(rescale=1./255,vertical_flip=True)
        
     train_generator = train_gen.flow_from_dataframe(dataframe = pd.DataFrame(data= {'filename': TrainDF['FULL_PATH'], 'class': TrainDF['Class']}), 
                                                     directory= None, image_data_generator= train_gen, x_col = 'filename', y_col= 'class', color_mode='rgb',
                                                     save_prefix='', target_size= targetsize, batch_size = batchsize, shuffle= True, class_mode= 'sparse', 
                                                     save_format='jpg', interpolation='nearest', validate_filenames = True, seed = seed)
     
     # V-A-L-I-D-A-T-I-O-N  (using the test data)
     print(" \n The samples for validation after each epoch are: ")       
     test_gen = ImageDataGenerator(rescale=1./255)  
                            
     valid_generator = test_gen.flow_from_dataframe(dataframe = pd.DataFrame(data= {'filename': PredDF['FULL_PATH'], 'class': PredDF['Class']}), 
                                                     directory= None, image_data_generator= train_gen, x_col = 'filename', y_col= 'class', color_mode='rgb',
                                                     save_prefix='', target_size = targetsize, batch_size = batchsize, shuffle= True, class_mode= 'sparse', 
                                                     save_format='jpg', interpolation='nearest', validate_filenames = True, seed = seed)
              
     # FOR P-R-E-D-I-C-T-I-O-N (using the test data, that was also used for validation) - THE NETWORK DOES NOT USE THIS DATA FOR TRAINING !! 
     # This is the same generator as the previous, with the only differences that, when testing, batch_size = 1 and Shuffle= False
     print(" \n The testing samples are: ")

     PredDF_generator = test_gen.flow_from_dataframe(dataframe = pd.DataFrame(data= {'filename': PredDF['FULL_PATH'], 'class': PredDF['Class']}), 
                                                    directory= None, image_data_generator= test_gen, x_col = 'filename', y_col= 'class', color_mode='rgb',
                                                    save_prefix='', target_size = targetsize, batch_size = 1, shuffle= False, class_mode= 'sparse',
                                                    save_format='jpg', interpolation='nearest', validate_filenames = True, seed = seed)                      
          
     # -------------------------------------------------------------------
     #                       LOAD THE MODEL                        
     #--------------------------------------------------------------------
     # Arguments: Set_Network(network_name, number_classes)
     model = Set_Network(network, len(train_generator.class_indices.keys()))
     
     # Epochs first
     epochs_first, epochs_total = 6, 10
     lr_1= 3e-4
     lr_2= 3e-5
    
     #_____________________COMPILE THE MODEL _____________________________
     #           1st compilation (to train only the added dense layers)
     model.compile(loss='sparse_categorical_crossentropy', optimizer= Adam(lr=lr_1,beta_1=0.9, beta_2=0.999,epsilon=1e-08, decay= lr_1/epochs_first), 
                                                                           metrics=['sparse_categorical_accuracy'])
                                   
     # Define the steps per epoch
     train_steps,  val_steps = len(train_generator), len(valid_generator)              
     
     # ___________________ T-R-A-I-N  the model___________________________
     #        (initialization of the weights in the added layers)
     print(" \n *** Train the {} with the basemodel layers untrained. Only the weights of the added Dense Layers are unfreezed.".format(network))
     
     Fit_history = model.fit_generator(generator = train_generator, validation_data = valid_generator, shuffle= True, steps_per_epoch = train_steps,
                                       validation_steps= val_steps, epochs=epochs_first, verbose=1, workers=4, max_queue_size = 20, 
                                       use_multiprocessing=False)
               
     # ____________________Fine-tuning the Base_Model ____________________
     # Unfreeze  convolutional layers from the current baseline network 
     if network=='ResNet50':
           net = 'resnet50'
           
           for layer in model.get_layer(net).layers[:165]:
               layer.trainable = False
               
           # Fine-tuning the last 10 layers
           for layer in model.get_layer(net).layers[165:]:
               layer.trainable = True
               
     elif network=='InceptionV3':
           net = 'inception_v3'
           for layer in model.get_layer(net).layers[:249]:
               layer.trainable = False
               
           #Fine-tuning the top 2 Inception blocks    
           for layer in model.get_layer(net).layers[249:]:
               layer.trainable = True
               
     elif network=='Xception':
           net = 'xception'
           for layer in model.get_layer(net).layers[:-16]:
               layer.trainable = False
           
           # Fine-tuning the last 16 layers
           for layer in model.get_layer(net).layers[-16:]:
               layer.trainable = True                              
          
     #__________________________ RE - COMPILE THE MODEL __________________________________
     #    2nd compilation (to train both the unfreezed Conv and the added dense layers)
     model.compile(loss='sparse_categorical_crossentropy', optimizer= Adam(lr=lr_2,beta_1=0.9, beta_2=0.999,epsilon=1e-08, decay=0.00005),
                   metrics=['sparse_categorical_accuracy']) 
              
     # Prepare a Callback to track the training and validation accuracy, and the training and validation loss            
     checkpoint = ModelCheckpoint(Results_Dirs + '\\' + os.path.basename(Tiles_Dir) + '_' + network + '_'+ experiment +'_Final' + 'Acc.h5', 
                                     monitor='sparse_categorical_accuracy', verbose=0, save_best_only = True)
     
     # Save the training resulted accuracies and losses, only for the epochs after the 2nd model compilation
     # The metrics from training after the 1st model compilation are only printed in the console during the first part of the training
     csv_logger= CSVLogger(os.path.basename(Tiles_Dir)+ '_' +  network + '_'+ experiment +'_Final' + ".log")
     
     # Reset data generators
     train_generator.reset()
     valid_generator.reset()
                   
     #___________________ RE - TRAIN  the model___________________________
     #         (Weights of the unfreezed layers will also be updated)
     print(" \n *** Train the chosen last Conv Layers of the {} and the added Dense Layers.".format(network))
     
     # Continue the re-training from the last epoch of the previous training
     FineTune_history = model.fit_generator(generator = train_generator, validation_data = valid_generator, shuffle= True, epochs=epochs_total, 
                                            initial_epoch = Fit_history.epoch[-1],steps_per_epoch = train_steps, validation_steps= val_steps,
                                            callbacks=[checkpoint,csv_logger], verbose=1, workers=4, max_queue_size = 20, use_multiprocessing=False)
     
     # Save the final model, where the weights from the last Conv chosen layers and the added ones were updated
     model.save(os.path.basename(Tiles_Dir)+ '_' +  network + '_'+ experiment+'_Final'+ '.h5')
     model.save_weights(os.path.basename(Tiles_Dir)+ '_'+ 'Weights_' + network +'_'+ experiment +'_Final' + '.h5')   
     
     
     # -------------------------- Plot the training and validation metrics from the last training  ---------------------
     # Note:  The following lines 258-272 are slightly adjusted and come from a plotting paradigm on 'tensorflow.org',
     # https://www.tensorflow.org/tutorials/images/classification?authuser=0&hl=zh-cn
     plt.style.use('seaborn-colorblind')
     
     fig, axs = plt.subplots(2)
     axs[0].plot(np.arange(0, epochs_total-epochs_first+1),  FineTune_history.history["loss"], "r",
                 np.arange(0, epochs_total-epochs_first+1),  FineTune_history.history["val_loss"], "-bo")
     axs[0].set_ylabel("Loss")
     axs[0].set_xlabel("Epochs")
     axs[0].set_title('Training and validation accuracy and loss', fontsize=12, y=1.109)
     plt.legend(["train","val"],loc="best")
         
     axs[1].plot(np.arange(0, epochs_total-epochs_first+1), FineTune_history.history["sparse_categorical_accuracy"], "r",
                 np.arange(0, epochs_total-epochs_first+1), FineTune_history.history["val_sparse_categorical_accuracy"], "-bo")
     axs[1].set_ylabel("Accuracy")
     axs[1].set_xlabel("Epochs")
     plt.legend(["train","val"],loc='best')
         
     fig.tight_layout()
     fig= plt.gcf()
     plt.show()
     plt.draw()
     fig.savefig(Results_Dirs + '\\' + network +'_'+ experiment +'_'+ '_Final' + ".png", dpi=1200, quality=95)
     plt.close()

     # ---------------------------------------------------------------------------------------------------------------------------
     #                                  PREDICTIONS FOR THE KEPT-OUT TEST DATA
     #----------------------------------------------------------------------------------------------------------------------------
     # Use the function 'Predict.py' to return the soft predictions 
     PredDF_generator.reset()
     print(" \n Predicting on the last fold of data:")
     
     # -------------------------------------------------------------------
     #                    P-R-E-D-I-C-T-I-N-G
     #--------------------------------------------------------------------
     
     # Leave the Idx empty
     Idx= ''
     
     # Predict
     Predictions_FoldData,  Predicted_Filtered = Predict(PredDF, model, network, 'TestData', PredDF_generator, Idx, Results_Dirs, cvtype, experiment, Project_Dir= Current_Dir).predictions()
     
     # Gather the correct predictions per patient and classes
     # Those where the highest predicted probabilities indeed belong to the True Label
     Correct_Predictions = Predicted_Filtered.drop(['Predicted','Position'],1)
     Correct_Predictions = Correct_Predictions.rename(columns={'True_Positives':'Probability'})
     Correct_Predictions = Correct_Predictions.sort_index(level=0)
     Patients = Correct_Predictions.index.unique()
     
     # This is a list where each entry is a Dataframe. Each dataframe has the results from one patient: 
     #  (patient image, Predicted_Label, True_Label, Predicted_Accuracy) 
     ListResultsPerPatient= []
     
     for els in list(Patients):
          ListResultsPerPatient.append(Correct_Predictions[Correct_Predictions.index==els])
     
     """
      Creat a dictionary with the class, the patient ids for this class, and
      the mean Probability from all of the images of each patient:
                                         
                                    patient_1 : Mean Prob
                 Class1             patient_2 : Mean Prob                 
                                        ...   :  ...
                                    patient_1 : Mean Prob
                 Class2             patient_2 : Mean Prob                  
     """      
     # Instantiate an empty dictionary
     MeanPntProb ={}
     
     # Iterate over each patient's results table to retrieve the Average Acc
     for order, patient in enumerate(ListResultsPerPatient):
          Index = patient.index.unique()[0]
          Subtype = patient['True_Labels'].unique()[0]
          MeanPntProb[Subtype,Index] = patient['Probability'].mean()
      
     # Sort the  MeanPntAcc based on the class name    
     Sorted_MeanPntProb = OrderedDict(sorted(MeanPntProb.items(), key=lambda val: val[0]))
     
     # This is the final Average Acc dataframe with two levels of indices (Class, patient id)
     Final_DF_AVERAGE_Prob_Pnts = pd.DataFrame(Sorted_MeanPntProb.values(), 
                                              pd.MultiIndex.from_frame(pd.DataFrame(Sorted_MeanPntProb.keys()),
                                                                       names=['Subtype', 'Patient'])) 
     Final_DF_AVERAGE_Prob_Pnts = Final_DF_AVERAGE_Prob_Pnts.rename(columns={0:'Average_Probability'})   

     # Save the dataframe to an *.xlsx file in the results folder
     Final_DF_AVERAGE_Prob_Pnts.to_excel(Results_Dirs + '\\' + os.path.basename(Tiles_Dir)+ '_'+ experiment + '_' + network + '_'+  '_Average_Prob_ClassPnt'+ cvtype + '.xlsx')
                                           
     # --------------------------------------------------------------------------------------------------------------------------
     #    The following plots are based on the image tiles, with the results not on the patient level but only on the class level
     #---------------------------------------------------------------------------------------------------------------------------
     # Plot the roc curve for each class for the current last fold used for predictions
     plot_roc(np.array(PredDF_generator.classes), Predictions_FoldData, PredDF, Idx, title='ROC Curve per class for predicting on the final Testing Data',
              plot_micro=False, plot_macro=False, classes_to_plot= None, ax=None, figsize=(14,7), cmap='tab20c', title_fontsize='x-large', text_fontsize='x-large')
     
     plt.savefig(Results_Dirs + '\\' + experiment + '_'+ network + 'Roc_Curves_' + DataType + '_Final' + '.png', dpi=1200, quality=95)  
     plt.close()
     
     # Plot the precision-recall curves for the current used fold
     plot_precision_recall_curve(np.array(PredDF_generator.classes), Predictions_FoldData, PredDF, Idx, title='Precision_Recall Curve per class for predicting on the final Testing Data',
                                 ax=None, figsize=(14,7), cmap='tab20c', title_fontsize='x-large', text_fontsize='large')        
     plt.savefig(Results_Dirs + '\\' + experiment + '_'+ network + 'Precision_Recall_Curves_' + DataType + '_Final' + '.png', dpi=1200, quality=95)  
     plt.close()
     
     
     #                            E V A L U A T I O N
     # Export the evaluation loss and accuracy, after evaluating the model on the Test Data
     Scores = model.evaluate_generator(PredDF_generator, steps = len(PredDF_generator))
     
     for the, metric in enumerate(model.metrics_names):
          print('{}: {}'.format(metric, Scores[the]))
          
     # Save the evaluation accuracy and loss     
     Scores_df = pd.DataFrame(data={'loss': Scores[0], 'Accuracy': Scores[1]}, index=['metrics'])      
     Scores_df.to_excel(Results_Dirs + '\\' + os.path.basename(Tiles_Dir)+ '_'+ experiment + '_' + network + '_'+  '_Eval_Scores'+ cvtype + '.xlsx')
#------------------------------------------------------------------------------
#                                  E N D
#------------------------------------------------------------------------------


