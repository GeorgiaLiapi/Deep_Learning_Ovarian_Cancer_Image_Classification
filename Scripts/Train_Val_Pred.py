# -*- coding: utf-8 -*-
"""
          Author: Georgia Liapi, Master Student in Systems Biology, Maastricht University
   Academic Year: 2019-2020
         Project: 'Deep learning-based prediction of molecular subtypes in human ovarian cancer.'
         Utility: Perform the training on (k-1) * data folds of the selected k-fold cross validation (3- or 5-fold) and generate predictions (patient level) on the kept-out data fold each time.
         
                  Re-training a pre-trained CNN, utilizing image tiles from Whole Slide Images, from three different datasets:
                  'OC (Ovarian Cancer Histological H&E images from the Simon Fraser University Dataset, (Köbel et al., 2010))
                  'TCGA-OV-DX'
                  'TCGA-OV-KR'
                  
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
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle  
from collections import OrderedDict, Counter
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import random, h5py, os, openpyxl
import ipykernel, sklearn
import PIL, math
import scikitplot 
 
# Import adjusted pre-existing functions used for this project 
from plot_roc import  plot_roc
from calc_roc_auc import roc_auc
from plot_metrics import plot_metrics
from precision_recall import plot_precision_recall_curve

# Import created functions for this project
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
OV_SFU = '..\\OV_PROJECT\\RESULTS\\HISTOLOGICAL_SUBTYPES'
OV_DX  = '..\\OV_PROJECT\\RESULTS\\TCGA_DX_MOLECULAR_SUBTYPES'
OV_KR  = '..\\OV_PROJECT\\RESULTS\\TCGA_KR_MOLECULAR_SUBTYPES'
Results_Dirs  = [OV_SFU, OV_DX, OV_KR] 


#------------------------------------------------------------------------------
#                   A     U N I V E R S A L       F U N C T I O N
#                    (Used for any of the 3 Image datasets) 
#------------------------------------------------------------------------------

def Train_Val_Pred(Tiles_Dir, folds, network, Results_Dirs, Project_Dir= Current_Dir, experiment='Hist', cvtype='_3CV_Sess_', DataType='last_fold'):

     global Train_Data, DataComb, Results             
          
     # Set parameters
     seed      = 42
     batchsize = 64
          
     if folds == 3:                    
          
          # Load the data - USE OF THE CLASS 'DataGen'
          Train_Data       = DataGen(Tiles_Dir, folds, 'train', Project_Dir).Generate()
     
          # ---------------- Create the combination of folds ------------------
          # 'DataComb' is a list of the k data tables (folds). 
          # Structure eg. DataComb[0] = [(concatenated and shuffled Fold_1 & Fold2),      Fold_0         ]
          #                             [            used for training             , used for predictions]
          DataComb = []
          
          for idx, els in enumerate(Train_Data):
               if idx == 0 :
                    # Train with [Fold_0,Fold_1] and predict for Fold_0
                    DataComb.append([shuffle(pd.concat([shuffle(Train_Data[idx +1]), shuffle(Train_Data[idx +2])])), shuffle(Train_Data[idx])])                 
               elif idx ==1:
                    # Train with [Fold_1,Fold_2]  and and predict for Fold_1
                    DataComb.append([shuffle(pd.concat([shuffle(Train_Data[idx -1]), shuffle(Train_Data[idx +1])])), shuffle(Train_Data[idx])])                    
               else:
                    # Train with [Fold_0, Fold_2] and predict for Fold_2
                    DataComb.append([shuffle(pd.concat([shuffle(Train_Data[idx -2]), shuffle(Train_Data[idx -1])])), shuffle(Train_Data[idx])])
                    
     elif   folds == 5:
          
          # Load the data - USE OF THE CLASS 'DataGen' ------------------------
          Train_Data       = DataGen(Tiles_Dir, folds, 'train', Project_Dir).Generate()
          
          # Create the combination of folds------------------------------------
          # 'DataComb' is a list of the k data tables (folds). 
          DataComb = []
          
          for idx, els in enumerate(Train_Data):
               
               if idx == 0 :
                    # Train with [Fold_1, Fold_2,Fold_3, Fold_4] and predict for Fold_0
                    DataComb.append([shuffle(pd.concat([shuffle(Train_Data[idx +1]), shuffle(Train_Data[idx +2]), shuffle(Train_Data[idx +3]),
                                                        shuffle(Train_Data[idx + 4])])), shuffle(Train_Data[idx])])                    
               elif idx ==1:
                    # Train with [Fold_0, Fold_1,Fold_2, Fold_3] and predict for Fold_1
                    DataComb.append([shuffle(pd.concat([shuffle(Train_Data[idx -1]), shuffle(Train_Data[idx +1]), shuffle(Train_Data[idx +2]),
                                                        shuffle(Train_Data[idx + 3])])), shuffle(Train_Data[idx])])               
               elif idx ==2:
                    # Train with [Fold_0, Fold_1,Fold_3, Fold_4] and predict for Fold_2
                    DataComb.append([shuffle(pd.concat([shuffle(Train_Data[idx -2]), shuffle(Train_Data[idx -1]), shuffle(Train_Data[idx +1]), 
                                                        shuffle(Train_Data[idx + 2])])), shuffle(Train_Data[idx])])                    
               elif idx ==3:
                    # Train with [Fold_0, Fold_1,Fold_2, Fold_4] and predict for Fold_3
                    DataComb.append([shuffle(pd.concat([shuffle(Train_Data[idx -3]), shuffle(Train_Data[idx -2]), shuffle(Train_Data[idx -1]), 
                                                        shuffle(Train_Data[idx + 1])])), shuffle(Train_Data[idx])])                   
               else:
                    # Train with [Fold_0, Fold_2,Fold_3, Fold_4] and predict for Fold_4
                    DataComb.append([shuffle(pd.concat([shuffle(Train_Data[idx -4]), shuffle(Train_Data[idx -3]), shuffle(Train_Data[idx -2]),
                                                        shuffle(Train_Data[idx -1])])), shuffle(Train_Data[idx])])
                                       
     """
     Instantiate the dictionary 'RocResults' to save the appended fpr, tpr and auc numbers per each of the current k predictions.
     Each key in the dictionary refers to the results of the corresponding (k) fold used for training) after
     all the training-predicting sessions of the k-fold cross validation.
     There will be as many keys as the number of the folds.
     """
     RocResults = dict()
     
     #-----------------------------------------------------------------------------------------------------------------------
     #                                **********   FROM HERE TO THE END  **********       
     #  This is a main loop that will run as many times as the number of the selected folds when calling the Train_Val_Pred()
     #-----------------------------------------------------------------------------------------------------------------------
     """
     Iterate through all of the combinations of training folds and predictions folds for the current k-fold cross-validation experiment.
     Eg. For 3-fold cross Validation, this loop will run totally 3 times. 
         Each time the model will be re-trained (with current combination of folds for training) and the current last fold will be used
         for both validation after each epoch and predictions; it is consider acceptable as in validation after each epoch, data are not
         used to used to update the gradient descent. The model, with its weights, per each of the 3 sessions of training will be separately saved.
     """      
     for Idx, i in enumerate(DataComb):
          
          """
          i[0]: 'Train'  data   
          i[1]: 'Prediction' data
          """
          
          # Clear the previous session to control for OOM Errors in every next run
          K.clear_session()
                    
          # Set the target size needed for the ImageDataGenerators
          if network == 'Xception' or network=='InceptionV3':               
                targetsize = (299,299)                                           
          else:               
                targetsize = (224,224)                

          # NOTE: The internal validation data are also used for predictions, as they are not actually used in the training to update the gradient descent
                
          # Set the validation (after each epoch) data
          # These are also the predictions data (THE SAME TABLE IS USED FOR BOTH)
          PredDF  = shuffle(shuffle(i[1]))
                 
          #  --------------------------------->     Re-sampling the Train Data     <------------------------------------------
          #  --------------------------------->  Same number of examples per Class <------------------------------------------
          
          # Dictionary of class keys and their number of examples -imbalanced
          Imbalanced_Classes_TrainFolds = Counter(i[0]['Class'])      
          print('\n The Training tiles per Class before re-sampling:{}'.format(Imbalanced_Classes_TrainFolds))
          
          # Find the number of examples in the minority class
          Minority_class_Train= np.array(list(Imbalanced_Classes_TrainFolds.values())).min()
          
          # Instantiate an new Train data table
          Train_resampled = pd.DataFrame()          
          
          # Select as many examples per Class as those in the minority class 
          # Essentially rows are selected from the primary Train data (the i[0]), but it cannot be controled that also each patient will contribute the same number of images
          
          # If the dataset is the KR, reduce the minority class tiles to 15000
          if Tiles_Dir==Dirs[2]:
               #Reduce the minority class image tiles even more (due to computational constraints)
               Minority_class_Train = 10000
               for theClass in list(Imbalanced_Classes_TrainFolds.keys()):
                    Train_resampled =  Train_resampled.append(i[0][i[0]['Class']== theClass].sample(n=Minority_class_Train,replace=False))
          else:
          # If the dataset is either the histological (SFU) or the TGCA-OV-DX     
               for theClass in list(Imbalanced_Classes_TrainFolds.keys()):
                    Train_resampled =  Train_resampled.append(i[0][i[0]['Class']== theClass].sample(n=Minority_class_Train,replace=False)) 
               
          # Dictionary of class keys and their number of examples-balanced
          Balanced_Classes_Train = Counter(Train_resampled['Class'])      
          print('\n The Training tiles per Class after Under-sampling:{}'.format(Balanced_Classes_Train))
          
          # Finally, very important! ---> SHUFFLE THE TRAIN DATA, to have a reasonable various number of classes later on each batch
          TrainDF = shuffle(shuffle(Train_resampled, random_state= 42))
                             
                            
          # ------------------------------------------------------------------------------------------------------------------                  
          #                                        Prepare the Image data generators
          #                               The data will flow from already created dataframes !!!
          #                   Dataframe Structure:  (Index:Patient Id, column A: ImageTiles Full Paths, column B: Class as String) 
          #-------------------------------------------------------------------------------------------------------------------
          # T-R-A-I-N-I-N-G 
          print(" \n The training samples are: ")   
          train_gen = ImageDataGenerator(rescale=1./255, vertical_flip=True)
             
          train_generator = train_gen.flow_from_dataframe(dataframe = pd.DataFrame(data= {'filename': TrainDF['FULL_PATH'], 'class': TrainDF['Class']}), 
                                                          directory= None, image_data_generator= train_gen, x_col = 'filename', y_col= 'class', color_mode='rgb',
                                                          save_prefix='', target_size= targetsize, batch_size = batchsize, shuffle= True, class_mode= 'sparse', 
                                                          save_format='jpg', interpolation='nearest', validate_filenames = True, seed = seed)
          
          # V-A-L-I-D-A-T-I-O-N  (using the last fold in each element of the DataComb)
          print(" \n The samples for validation after each epoch are: ")       
          test_gen = ImageDataGenerator(rescale=1./255)  
                                 
          valid_generator = test_gen.flow_from_dataframe(dataframe = pd.DataFrame(data= {'filename': PredDF['FULL_PATH'], 'class': PredDF['Class']}), 
                                                          directory= None, image_data_generator= train_gen, x_col = 'filename', y_col= 'class', color_mode='rgb',
                                                          save_prefix='', target_size = targetsize, batch_size = batchsize, shuffle= True, class_mode= 'sparse', 
                                                          save_format='jpg', interpolation='nearest', validate_filenames = True, seed = seed)
                   
          # FOR P-R-E-D-I-C-T-I-O-N (using the last fold in each element of the DataComb, that was also used for validation) - THE NETWORK DOES NOT USE THIS DATA FOR TRAINING !! 
          # This is the same generator as the previous, with the only difference that when predicting, batch_size = 1 and Shuffle is False, to preserve the position of the images
          # and their patients' indices, when matching predictions to patient anonymous ids
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
          epochs_first, epochs_total = 7, 11
          
          # Learning rate for training only the added layers
          lr_1= 3e-4
          
          # Learning rate for training the last chosen layers from the base model
          # plus (re-training) the added layers
          lr_2= 3e-5
          
          #_____________________COMPILE THE MODEL _____________________________
          #           1st compilation to train only the added dense layers !!
          model.compile(loss='sparse_categorical_crossentropy', optimizer= Adam(lr=lr_1,beta_1=0.9, beta_2=0.999,epsilon=1e-08, decay= lr_1/epochs_first), 
                                                                                metrics=['sparse_categorical_accuracy'])
                                        
          # Define the steps per epoch for each data generator
          train_steps,  val_steps = len(train_generator), len(valid_generator)              
          
          # ___________________ T-R-A-I-N  the model___________________________
          #        (random initialization of the weights in the added layers)
          print(" \n *** Train the {} with the basemodel layers untrained. Only the weights of the added Dense Layers are unfreezed.".format(network))
          
          Fit_history = model.fit_generator(generator = train_generator, validation_data = valid_generator, shuffle= True, steps_per_epoch = train_steps,
                                            validation_steps= val_steps, epochs=epochs_first, verbose=1, workers=4, max_queue_size = 20, 
                                            use_multiprocessing=False)
                    
          # ____________________Fine-tuning some of the last layers in the Base_Model ____________________
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
          #    2nd compilation to train the unfreezed base model layers and the added dense layers !!
          model.compile(loss='sparse_categorical_crossentropy', optimizer= Adam(lr=lr_2,beta_1=0.9, beta_2=0.999,epsilon=1e-08, decay=0.00005),
                        metrics=['sparse_categorical_accuracy']) 
                   
          # Prepare a Callback to track the training and validation accuracy, and the training and validation loss               
          checkpoint = ModelCheckpoint(Results_Dirs + '\\' + os.path.basename(Tiles_Dir) + '_' + network + '_'+ experiment + str(folds) +'_CV_fold' + str(Idx) + 'Acc.h5', 
                                          monitor='sparse_categorical_accuracy', verbose=0, save_best_only = True)
          
          #Save the training resulted accuracies and losses, only for the epochs after the 2nd model compilation
          # The metrics from training after the 1st model compilation are only printed in the console during the first part of the training
          csv_logger= CSVLogger(os.path.basename(Tiles_Dir)+ '_' +  network + '_'+ experiment + str(folds) +'_CV_fold' + str(Idx)+ ".log")
          
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
          model.save(os.path.basename(Tiles_Dir)+ '_' +  network + '_'+ experiment + str(folds) +'_CV_fold' + str(Idx) + '.h5')
          model.save_weights(os.path.basename(Tiles_Dir)+ '_'+ 'Weights_' + network +'_'+ experiment + str(folds) +'_CV_fold' + str(Idx) + '.h5')   
          
          
          # --------------------------------------------------- Plot the training and validation metrics from the training after the 2nd compilation -----------------------
          plot_metrics(FineTune_history.history["loss"], FineTune_history.history["val_loss"], FineTune_history.history["sparse_categorical_accuracy"], 
                       FineTune_history.history["val_sparse_categorical_accuracy"], epochs_total-epochs_first+1, Results_Dirs, network, cvtype, Idx, 
                       'Accuracy and Loss for the training session in ' + str(folds) +'_CV' +'(session'+ str(Idx+1)+')', experiment)      
    
          # ---------------------------------------------------------------------------------------------------------------------------------------------
          #                                                        PREDICTIONS FOR THE KEPT-OUT FOLD
          #----------------------------------------------------------------------------------------------------------------------------------------------
          # Use the function 'Predict.py' to return the soft predictions 
          
          print(' \n               *** Fold No{} is now used for predictions ***'.format(Idx+1))
          
          PredDF_generator.reset()
          
          # -------------------------------------------------------------------
          #                    P-R-E-D-I-C-T-I-N-G
          #--------------------------------------------------------------------
          Predictions_FoldData,  Predicted_Filtered = Predict(PredDF, model, network, 'last_fold', PredDF_generator, Idx, Results_Dirs, cvtype, experiment, Project_Dir= Current_Dir).predictions()
          
          # Gather the correct predictions per patient and classes
          # Those where the highest predicted probabilities indeed belong to the True Label
          Correct_Predictions = Predicted_Filtered.drop(['Predicted','Position'],1)
          Correct_Predictions = Correct_Predictions.rename(columns={'True_Positives':'Accuracy'})
          Correct_Predictions = Correct_Predictions.sort_index(level=0)
          Patients = Correct_Predictions.index.unique()
          
          # This is a list where each entry is a Dataframe. Each dataframe has the results from each patient as: 
          #  (patient image, Predicted_Label, True_Label, Predicted_Accuracy) 
          ListResultsPerPatient= []
          
          for els in list(Patients):
               ListResultsPerPatient.append(Correct_Predictions[Correct_Predictions.index==els])
          
          """
           Creat a dictionary with the class, the patient ids for this class, and
           the mean accuracy from all of the images of each patient:
                                              
                                         patient_1 : Mean Probability
                      Class1             patient_2 : Mean Probability                  
                                             ...   :  ...
                                         patient_1 : Mean Probability
                      Class2             patient_2 : Mean Probability                   
          """      
          # Instantiate an empty dictionary
          MeanPntProb ={}
          
          # Iterate over each patient's results table to retrieve the Average Acc
          for order, patient in enumerate(ListResultsPerPatient):
               Index = patient.index.unique()[0]
               Subtype = patient['True_Labels'].unique()[0]
               MeanPntProb[Subtype,Index] = patient['Accuracy'].mean()
           
          # Sort the  MeanPntAcc based on the class name    
          Sorted_MeanPntProb = OrderedDict(sorted(MeanPntProb.items(), key=lambda val: val[0]))
          
          # This is the final Average Acc dataframe with two levels of indices (Class, patient id)
          Final_DF_AVERAGE_Prob_Pnts = pd.DataFrame(Sorted_MeanPntProb.values(), 
                                                   pd.MultiIndex.from_frame(pd.DataFrame(Sorted_MeanPntProb.keys()),
                                                                            names=['Subtype', 'Patient'])) 
          Final_DF_AVERAGE_Prob_Pnts = Final_DF_AVERAGE_Prob_Pnts.rename(columns={0:'Average_Probability'})   

          # Save the dataframe to an *.xlsx file in the results folder
          Final_DF_AVERAGE_Prob_Pnts.to_excel(Results_Dirs + '\\' + os.path.basename(Tiles_Dir)+ '_'+ experiment + '_' + network + '_'+  '_Average_Acc_ClassPnt'+ cvtype + str(Idx) + '.xlsx')
                                                
          # --------------------------------------------------------------------------------------------------------------------------
          #    The following plots are based on the image tiles, with the results not on the patient level but only on the class level
          #---------------------------------------------------------------------------------------------------------------------------
          # Plot the roc curve for each class for the current last fold used for predictions
          plot_roc(np.array(PredDF_generator.classes), Predictions_FoldData, PredDF, Idx, title='ROC Curve per class for predicting on the Fold_{} \n ({}-fold Cross Validation)'.format(str(Idx+1),
                   str(folds)), plot_micro=False, plot_macro=False, classes_to_plot= None, ax=None, figsize=(14,7), cmap='tab20c', title_fontsize='x-large', text_fontsize='x-large')
          
          plt.savefig(Results_Dirs + '\\' + experiment + '_'+ network + 'Roc_Curves_' + DataType + str(folds) +'_CV_fold' + str(Idx) + '.png', dpi=1200, quality=95)  
          plt.close()
          
          # Plot the precision-recall curves for the current used fold
          plot_precision_recall_curve(np.array(PredDF_generator.classes), Predictions_FoldData, PredDF, Idx, title='Precision_Recall Curve per class for predicting on the Fold_{} \n ({}-fold Cross Validation)'.format(str(Idx+1), str(folds)),
                                      ax=None, figsize=(14,7), cmap='tab20c', title_fontsize='x-large', text_fontsize='large')        
          plt.savefig(Results_Dirs + '\\' + experiment + '_'+ network + 'Precision_Recall_Curves_' + DataType + str(folds) +'_CV_fold' + str(Idx) + '.png', dpi=1200, quality=95)  
          plt.close()
          
          # ---------------------------------------------------------------------------------------------------------------------------
          #         Generate the false positive rate, the true positive rates, as well as the auc numbers per fold for all classes 
          # ---------------------------------------------------------------------------------------------------------------------------
          if DataType == 'last_fold':
     
               RocResults[Idx] = roc_auc(np.array(PredDF_generator.classes), Predictions_FoldData, Idx, classes_to_plot= None)
          
          # Clear the session again to prevent OOM errors
          K.clear_session()
                   
          #--------------------------------------------------------------------
          #                  E N D    O F   M A I N    L O O P
          #--------------------------------------------------------------------
     # Save the false positive and true positive rates, as well as the auc numbers per fold for all classes as excel file 
     for indx, items in enumerate(RocResults.items()):
          
          # Save the results 
          pd.DataFrame(RocResults[indx], 
                       index = ['fpr','tpr','auc']).to_excel(Results_Dirs + '\\' + os.path.basename(Tiles_Dir) + '_' + network + '_'+ experiment + 'FprTprAuc'+ cvtype + str(indx) + '.xlsx')
#------------------------------------------------------------------------------
#                                  E N D
#------------------------------------------------------------------------------
