import numpy as np
import os
import tensorflow as tf
import flwr as fl
import tensorflow.keras.layers
from keras.models import load_model
import keras.backend as K
import time
import math
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import pickle as pkl
import matplotlib.pyplot as plt
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
h=math.log(2)
num_classes = 10
input_shape = (28, 28, 1)
batch_size0=0

s0,s1,s2,s3,s4,s5,s6,s7,s8,s9=np.zeros([10]),np.zeros([10]),np.zeros([10]),np.zeros([10]),np.zeros([10]),np.zeros([10]),np.zeros([10]),np.zeros([10]),np.zeros([10]),np.zeros([10])
wu=0.0    

Ml,Ma,Mb,Mx=np.zeros((10,10)),np.zeros((10,10)),np.zeros((10,10)),np.zeros((10,10))

loss_IRM=0
current_round=0

def softmax(x, temperature=2):
     return np.exp(x/temperature)/sum(np.exp(x/temperature))
 
def kl_d(p, q): 
      return tf.reduce_sum(p * np.log(p/q))

with open("traind2.pkl", "rb") as f:
     x_train, y_train = pkl.load(f)
with open("testd2.pkl", "rb") as f1:
     x_test, y_test = pkl.load(f1)


def loss(act,pred):
    loss_c=(K.sparse_categorical_crossentropy(act, pred, from_logits=True))
    # loss_c=tf.keras.losses.mean_squared_error(act, pred, from_logits=True)
    tf.cast(loss_c, tf.float32)
    tf.cast(loss_IRM, tf.float32)
    a=loss_c+loss_IRM
    warm=math.exp( -5*((1-current_round)/30) )
    war=tf.convert_to_tensor(warm)
    wi=tf.cast(war, dtype='float32')
    l=wi*(a)
    return l


# Define Flower client
class My_Client(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        

    def get_properties(self, config):
        """Get properties of client."""
        raise Exception("Not implemented")

    def get_parameters(self):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""
        
        global current_round,batch_size0
        batch_size0=64
        current_round=config["current_round"]
        if(current_round<=2):
            time.sleep(10)
        
     

        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        class number:
                value = -1
                def __new__(cls):
                    cls.value += 1
                    return cls.value
        class number2:
            m=64
            def __new__(cls):
                
                return cls.m
            
        # Train the model using hyperparameters from config
        class CustomCallback(tf.keras.callbacks.Callback):
       
            
            def on_train_batch_end(self,batch,logs=None):
                np.seterr(divide='ignore', invalid='ignore')
                j=number()
                # lim=number1()
                bs=number2()     
              

                global x_train,y_train,x_test,y_test
                global loss_IRM
                global Ml,Ma,Mb,Mx
                global s0,s1,s2,s3,s4,s5,s6,s7,s8,s9
                
            
                x_trainm=x_train[bs*(j):bs*(j+1)]
                pu=self.model.predict_on_batch(x_trainm)                   
                pudf = pd.DataFrame(pu)
                
                yu=np.argmax(pu,axis=1)             
                qu=np.stack([self.model(x_trainm,training=True)
                                    for sample in range(8)])        
                qu_m=qu.mean(axis=0)
                
                wu=(-(qu_m*np.log2(qu_m)))
     
                wu[np.isnan(wu)] = 100.0
         
                wudf = pd.DataFrame(wu)
               
                Ma=np.genfromtxt('Ma.csv', delimiter=',')  
                Mb=np.genfromtxt('Mb.csv', delimiter=',')
                Ml=((Ma+Mb)/2)
     
                sl0=Ml[:,0]
                sl1=Ml[:,1]
                sl2=Ml[:,2]
                sl3=Ml[:,3]
                sl4=Ml[:,4]
                sl5=Ml[:,5]
                sl6=Ml[:,6]
                sl7=Ml[:,7]
                sl8=Ml[:,8]
                sl9=Ml[:,9]
                
                c0_indexes = np.where((yu == 0 ))
                c1_indexes = np.where((yu == 1 ))
                c2_indexes = np.where((yu == 2 ))
                c3_indexes = np.where((yu == 3 ))
                c4_indexes = np.where((yu == 4 ))
                c5_indexes = np.where((yu == 5 ))
                c6_indexes = np.where((yu == 6 ))
                c7_indexes = np.where((yu == 7 ))
                c8_indexes = np.where((yu == 8 ))
                c9_indexes = np.where((yu == 9 ))
                
                wudf0,wudf1,wudf2,wudf3,wudf4,wudf5,wudf6,wudf7,wudf8,wudf9=pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
                
                wf0,wf1,wf2,wf3,wf4,wf5,wf6,wf7,wf8,wf9=pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
                
                pi0,pi1,pi2,pi3,pi4,pi5,pi6,pi7,pi8,pi9=pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()

                wi0,wi1,wi2,wi3,wi4,wi5,wi6,wi7,wi8,wi9=[],[],[],[],[],[],[],[],[],[]
                
                v0,v1,v2,v3,v4,v5,v6,v7,v8,v9=0,0,0,0,0,0,0,0,0,0
       
                loss0,loss1,loss2,loss3,loss4,loss5,loss6,loss7,loss8,loss9=0,0,0,0,0,0,0,0,0,0

                
                if (np.array(c0_indexes).size!=0):
            
                    wudf0=wudf.iloc[c0_indexes]
                    wf0=wudf0[wudf0[0] < h]
                    wf0=wf0[0]
                    wi0=wf0.index.to_numpy()
                    pi0=pudf.iloc[wi0]
                    v0=(pi0.to_numpy()).sum(axis=0)/len(pi0)
                    
                    s0=softmax(v0,temperature=2)
                    kl0=kl_d(s0,sl0)
                    kl0l=kl_d(sl0,s0)
                    if(~np.isnan(kl0)):
                        if(~np.isnan(kl0l)):
                            loss0=kl0+kl0l
                    

                if (np.array(c1_indexes).size!=0):
                
                    wudf1=wudf.iloc[c1_indexes]
                    wf1=wudf1[wudf1[1] < h]
                    wf1=wf1[1]
                    wi1=wf1.index.to_numpy()
                    pi1=pudf.iloc[wi1]
                    v1=(pi1.to_numpy()).sum(axis=0)/len(pi1)
                   
                    s1=softmax(v1,temperature=2)
                    kl1=kl_d(s1,sl1)
                    kl1l=kl_d(sl1,s1)
                    if(~np.isnan(kl1)):
                        if(~np.isnan(kl1l)):
                            loss1=kl1+kl1l
     
                if (np.array(c2_indexes).size!=0):
              
                    wudf2=wudf.iloc[c2_indexes]
                    wf2=wudf2[wudf2[2] < h]
                    wf2=wf2[2]
                    wi2=wf2.index.to_numpy()
                    pi2=pudf.iloc[wi2]
                    v2=(pi2.to_numpy()).sum(axis=0)/len(pi2)
                 
                    s2=softmax(v2,temperature=2)
                    kl2=kl_d(s2,sl2)
                    kl2l=kl_d(sl2,s2)
                    if(~np.isnan(kl2)):
                        if(~np.isnan(kl2l)):
                            loss2=kl2+kl2l
           
                if (np.array(c3_indexes).size!=0):
          
                   wudf3=wudf.iloc[c3_indexes]
                   wf3=wudf3[wudf3[3] < h]
                   wf3=wf3[3]
                   wi3=wf3.index.to_numpy()
                   pi3=pudf.iloc[wi3]
                   v3=(pi3.to_numpy()).sum(axis=0)/len(pi3)
                 
                   s3=softmax(v3,temperature=2)
                   kl3=kl_d(s3,sl3)
                   kl3l=kl_d(sl3,s3)
                   if(~np.isnan(kl3)):
                        if(~np.isnan(kl3l)):
                            loss3=kl3+kl3l
         
                if (np.array(c4_indexes).size!=0):
      
                   wudf4=wudf.iloc[c4_indexes]
                   wf4=wudf4[wudf4[4] < h]
                   wf4=wf4[4]
                   wi4=wf4.index.to_numpy()
                   pi4=pudf.iloc[wi4]
                   v4=(pi4.to_numpy()).sum(axis=0)/len(pi4)
            
                   s4=softmax(v4,temperature=2)
                   kl4=kl_d(s4,sl4)
                   kl4l=kl_d(sl4,s4)
                   if(~np.isnan(kl4)):
                        if(~np.isnan(kl4l)):
                            loss4=kl4+kl4l
                   
                if (np.array(c5_indexes).size!=0):
         
                      wudf5=wudf.iloc[c5_indexes]
                      wf5=wudf5[wudf5[5] < h]
                      wf5=wf5[5]
                      wi5=wf5.index.to_numpy()
                      pi5=pudf.iloc[wi5]
                      v5=(pi5.to_numpy()).sum(axis=0)/len(pi5)
           
                      s5=softmax(v5,temperature=2)
                      kl5=kl_d(s5,sl5)
                      kl5l=kl_d(sl5,s5)
                      if(~np.isnan(kl5)):
                        if(~np.isnan(kl5l)):
                            loss5=kl5+kl5l
       
                if (np.array(c6_indexes).size!=0):
                  
                      wudf6=wudf.iloc[c6_indexes]
                      wf6=wudf6[wudf6[6] < h]
                      wf6=wf6[6]
                      wi6=wf6.index.to_numpy()
                      pi6=pudf.iloc[wi6]
                      v6=(pi6.to_numpy()).sum(axis=0)/len(pi6)
         
                      s6=softmax(v6,temperature=2)
                      kl6=kl_d(s6,sl6)
                      kl6l=kl_d(sl6,s6)
                      if(~np.isnan(kl6)):
                        if(~np.isnan(kl6l)):
                            loss6=kl6+kl6l
                      
                if (np.array(c7_indexes).size!=0):
                
                    wudf7=wudf.iloc[c7_indexes]
                    wf7=wudf7[wudf7[7] < h]
                    wf7=wf7[7]
                    wi7=wf7.index.to_numpy()
                    pi7=pudf.iloc[wi7]
                    v7=(pi7.to_numpy()).sum(axis=0)/len(pi7)
      
                    s7=softmax(v7,temperature=2)
                    kl7=kl_d(s7,sl7)
                    kl7l=kl_d(sl7,s7)
                    if(~np.isnan(kl7)):
                        if(~np.isnan(kl7l)):
                            loss7=kl7+kl7l
     
                if (np.array(c8_indexes).size!=0):
              
                    wudf8=wudf.iloc[c8_indexes]
                    wf8=wudf8[wudf8[8] < h]
                    wf8=wf8[8]
                    wi8=wf8.index.to_numpy()
                    pi8=pudf.iloc[wi8]
                    v8=(pi8.to_numpy()).sum(axis=0)/len(pi8)
   
                    s8=softmax(v8,temperature=2)
                    kl8=kl_d(s8,sl8)
                    kl8l=kl_d(sl8,s8)
                    if(~np.isnan(kl8)):
                        if(~np.isnan(kl8l)):
                            loss8=kl8+kl8l
           
                if (np.array(c0_indexes).size!=0):
          
                   wudf9=wudf.iloc[c9_indexes]
                   wf9=wudf9[wudf9[9] < h]
                   wf9=wf9[9]
                   wi9=wf9.index.to_numpy()
                   pi9=pudf.iloc[wi9]
                   v9=(pi9.to_numpy()).sum(axis=0)/len(pi9)
    
                   s9=softmax(v9,temperature=2)
                   kl9=kl_d(s9,sl9)
                   kl9l=kl_d(sl9,s9)
                   if(~np.isnan(kl9)):
                        if(~np.isnan(kl9l)):
                            loss9=kl9+kl9l

                Mx=np.column_stack((s0,s1,s2,s3,s4,s5,s6,s7,s8,s9))
                np.savetxt("Md.csv", Mx, delimiter=",")

                 
                loss_IR=(loss0+loss1+loss2+loss3+loss4+loss5+loss6+loss7+loss8+loss9)/10
                loss_IRM = tf.convert_to_tensor(loss_IR)
                loss_IRM = tf.cast(loss_IRM, tf.float32)
                np.seterr(divide='ignore', invalid='ignore')
                # print('loss_IRM=',loss_IRM.numpy())
    
        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size,
            epochs,
            callbacks=[CustomCallback()],
            validation_split=0.1,
        )
        hist_df = pd.DataFrame(history.history) 

        hist_csv_file = 'historyd.csv'
        with open(hist_csv_file, mode='a') as f:
            hist_df.to_csv(f)

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["acc"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_acc"][0],
            
        }
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        # steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test,batch_size=64)
        num_examples_test = len(self.x_test)

        prediction =(self.model.predict(self.x_test,batch_size=64))
        y_tests=softmax(prediction,temperature=1)
        y_tests=np.argmax(y_tests,axis=-1)
        # accuracy = metrics.accuracy_score(self.y_test, y_tests)
  
        # # wrong_predictions = self.x_test[y_tests != self.y_test]
        indices = [i for i,v in enumerate(y_tests) if y_tests[i]!=self.y_test[i]]
        subset_of_wrongly_predicted = [i for i in indices ]
        count=0
        for i in subset_of_wrongly_predicted:
            if(current_round>=50):
                 if(count<=100):
                    count+=1
                
                    plt.imshow(self.x_test[i][:,:,0])
                    plt.savefig("d" +str(self.y_test[i])+"-"+str(y_tests[i])+"-"+str(i)+".png", dpi=300)
        
        
        return loss, num_examples_test, {"accuracy": accuracy}


def maina() -> None:

    
    modeli = load_model('modelZ.h5',compile=False)
    modeli.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.99),
                  loss=loss, metrics=['acc'])
    
    y_train1 = modeli.predict(x_train)

    # Convert predictions classes to one hot vectors 
    y_train1 = np.argmax(y_train1,axis = 1) 

      # Load and compile Keras model
    modelm=tf.keras.models.Sequential([
        tf.keras.layers.RandomRotation(0.3,input_shape=input_shape),
        tf.keras.layers.RandomZoom(0.1),
        modeli,
        
        
        ])
    del modeli
    modelm.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.99),
                  loss=loss,metrics=['acc'])


    # Start Flower client
    client = My_Client(modelm, x_train, y_train1, x_test, y_test)

    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=client,
        #root_certificates=Path(".cache/certificates/ca.crt").read_bytes(),
    )

if __name__ == "__main__":
    maina()
