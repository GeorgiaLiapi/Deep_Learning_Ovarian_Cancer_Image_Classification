"""
    || DISCLAIMER: THE FOLLOWING FUNCTION'S CONTENT WAS NOT A PRODUCT OF THE WORK OF GEORGIA LIAPI! || 
    || The following function's content is a plotting paradigm belonging to the 'tensorflow.org', https://www.tensorflow.org/tutorials/images/classification?authuser=0&hl=zh-cn
    
          User: Georgia Liapi, Master Student in Systems Biology, Maastricht University
 Academic Year: 2019-2020
       Project: 'Deep learning-based prediction of molecular subtypes in human ovarian cancer'
       Utility: This function will generate the plots with the training and validation metrics on the current Train_Val_Pred.py or
                Train_whole_Test.py experiment
                (Only for the number of epochs when training both the part of the base model and the added layers)
                
          Note:  The following lines 22-41 come from a plotting paradigm on 'tensorflow.org', https://www.tensorflow.org/tutorials/images/classification?authuser=0&hl=zh-cn
"""

# Import the required libraries
import matplotlib.pyplot as plt
import numpy as np

def plot_metrics(t_loss, v_loss, t_acc, v_acc, epochs, Results_Dir, network, cvtype, i, title, experiment):
     
     # ------ Plot the training and validation metrics ---------
     plt.style.use('seaborn-colorblind')
     
     fig, axs = plt.subplots(2)
     axs[0].plot(np.arange(0, epochs),  t_loss, "r",
                 np.arange(0, epochs),  v_loss, "-bo")
     axs[0].set_ylabel("Loss")
     axs[0].set_xlabel("Epochs")
     axs[0].set_title(title, fontsize=12, y=1.109)
     plt.legend(["train","val"],loc="upper right")
         
     axs[1].plot(np.arange(0, epochs), t_acc, "r",
                 np.arange(0, epochs), v_acc, "-bo")
     axs[1].set_ylabel("Accuracy")
     axs[1].set_xlabel("Epochs")
     plt.legend(["train","val"],loc='center right')
         
     fig.tight_layout()
     fig= plt.gcf()
     plt.show()
     plt.draw()
     fig.savefig(Results_Dir + '\\' + network +'_'+ experiment +'_'+  cvtype + str(i) + ".png", dpi=1200, quality=95)
     plt.close()