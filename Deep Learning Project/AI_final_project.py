import numpy as np                                                
import pandas as pd                                                
import matplotlib.pyplot as plt                                    
import seaborn as sns                                             
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dropout
from keras.optimizers import Adam
'''
Code: flow
1. read data and preprocess data. Check null, feature engineer and scale the data 
2. experimental plan for the model. I use 4 functions to build 4 different model 
and validate my model use 3 folds cv
3. run final model use callbacks 
4. load weight and see the test accuracy 
'''

'''
Variable:
avocado: the data
train: train data
test: test data
epoch: number of epoch
batch: number of batch
'''

avocado = pd.read_csv("avocado.csv") # read data
avocado.head() # check head
avocado.shape # check shape
avocado.info() # check null 
avocado.describe() # check outliers
avocado.drop("Unnamed: 0", axis=1,inplace=True) # drop index
avocado.rename(columns={'4046':'Small HASS sold',
                          '4225':'Large HASS sold',
                          '4770':'XLarge HASS sold'}, 
                 inplace=True) # change name

avocado.type.unique() # check number of category


avocado = pd.get_dummies(avocado, columns=['type'],drop_first=True) # transfer the target variabele 
avocado.head()


print("Organic ",avocado[avocado["type_organic"]==1]["Date"].count())
print("conventional ",avocado[avocado["type_organic"]==0]["Date"].count())

#check if the data balanced or not unbalanced

sns.heatmap(avocado.corr()) check correlation
le = LabelEncoder() 
avocado['region'] = le.fit_transform(avocado['region']) # encode region
avocado.head()
avocado.region.unique()

avocado['month']=pd.Series(list(map(lambda x: x.split('-')[1],list(avocado.Date)))) # extract month from date 

avocado.drop(['Date'],inplace=True,axis=1)


avocado.drop(['Total Volume','Total Bags'],inplace=True,axis=1) # drop unused column


scaler = StandardScaler()
scaler.fit(avocado.drop(['month','type_organic', 'year'],axis=1)) # scale the data 
scaled_features = scaler.transform(avocado.drop(['month','type_organic', 'year'],axis=1))

s=pd.DataFrame(scaled_features)
s.columns=['AveragePrice', 'Small HASS sold', 'Large HASS sold',
       'XLarge HASS sold', 'Small Bags', 'Large Bags', 'XLarge Bags','region']


avocado.columns


df=pd.concat([s,avocado.loc[:,['year',
        'type_organic', 'month']]],axis=1) # concat all the data 

temp={j:i for i,j in enumerate(list(df.year.unique()))} # map year in 0,1,2,3 
df.year=df.year.map(temp)
df.month.unique()
temp2={'01':0,'02':0,'03':0,'04':1,'05':1,'06':1,'07':2,'08':2,'09':2,'10':3,'11':3,'12':3}
temp2
df.month=df.month.map(temp2) # map month in season



df.info()
x=df.drop(['type_organic'],axis=1)
y=df['type_organic']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0) # create train test data 

# build train test data 
train=pd.DataFrame(X_train)
train['type']=y_train
test=pd.DataFrame(X_test)
test['type']=y_test
test.to_csv('test.csv')
train.to_csv('train.csv')
train_x=train.iloc[:,0:10]
train_y=train['type']


'''
First model, no hidden units 
'''
def create_model():    
    model = Sequential()
    model.add(Dense(64, kernel_initializer='normal',input_dim = train.shape[1]-1, activation='relu'))
    model.add(Dense(1, kernel_initializer='normal',activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    model.summary()
    return model

'''
Cross validation 
'''
n_split = 3
lst=[]# collect val accuracy 
for train_index, test_index in KFold(n_splits=n_split,shuffle = True).split(train):
    train_x=train.iloc[:,0:10]
    train_y=train['type']
    x_train,x_val=train_x.iloc[train_index,:],train_x.iloc[test_index,:]
    y_train,y_val=train_y.iloc[train_index],train_y.iloc[test_index]
    
    model = create_model()
    history = model.fit(x_train, y_train,batch_size = 10, epochs =50,verbose = 0,validation_data=(x_val,y_val))
    a=model.evaluate(x_val,y_val)[1]
    lst.append(a)
    epochs = range(len(history.history['acc']))
    plt.plot(history.history["acc"],label="acc")
    plt.plot(history.history['val_acc'], label = 'val_acc')
    plt.ylabel('acc value')
    plt.xlabel('epoch')
    plt.legend(loc="upper right")
    plt.show()

'''
Second model, with hidden layers and more hidden units 
'''
def create_model1():  
    # add 3 hidden layer with 128 hidden unit   
    model = Sequential()
    model.add(Dense(64, kernel_initializer='normal',input_dim = train.shape[1]-1, activation='relu'))
    model.add(Dense(128, kernel_initializer='normal',input_dim = train.shape[1]-1, activation='relu'))
    model.add(Dense(128, kernel_initializer='normal',input_dim = train.shape[1]-1, activation='relu'))
    model.add(Dense(128, kernel_initializer='normal',input_dim = train.shape[1]-1, activation='relu'))
    model.add(Dense(1, kernel_initializer='normal',activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    model.summary()
    return model

n_split = 3
lst1=[]# collect val accuracy 
for train_index, test_index in KFold(n_splits=n_split,shuffle = True).split(train):
    train_x=train.iloc[:,0:10]
    train_y=train['type']
    x_train,x_val=train_x.iloc[train_index,:],train_x.iloc[test_index,:]
    y_train,y_val=train_y.iloc[train_index],train_y.iloc[test_index]
    
    model = create_model1()
    history = model.fit(x_train,np.array(y_train).reshape(len(y_train),),batch_size = 10, epochs =100,verbose = 0,validation_data=(x_val,np.array(y_val).reshape(len(y_val),)))
    a=model.evaluate(x_val,y_val)[1]
    lst1.append(a)
    epochs = range(len(history.history['acc']))
    plt.plot(history.history["acc"],label="acc")
    plt.plot(history.history['val_acc'], label = 'val_acc')
    plt.ylabel('acc value')
    plt.xlabel('epoch')
    plt.legend(loc="upper right")
    plt.show()

'''
Third model, change the activation function from relu to tanh 
'''
def create_model2():  
    # add 3 hidden layer with 128 hidden unit  with tanh  
    model = Sequential()
    model.add(Dense(64, kernel_initializer='normal',input_dim = train.shape[1]-1, activation='tanh'))
    model.add(Dense(128, kernel_initializer='normal',input_dim = train.shape[1]-1, activation='tanh'))
    model.add(Dense(128, kernel_initializer='normal',input_dim = train.shape[1]-1, activation='tanh'))
    model.add(Dense(128, kernel_initializer='normal',input_dim = train.shape[1]-1, activation='tanh'))
    model.add(Dense(1, kernel_initializer='normal',activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    model.summary()
    return model

n_split = 3
lst2=[]# collect val accuracy 
for train_index, test_index in KFold(n_splits=n_split,shuffle = True).split(train):
    train_x=train.iloc[:,0:10]
    train_y=train['type']
    x_train,x_val=train_x.iloc[train_index,:],train_x.iloc[test_index,:]
    y_train,y_val=train_y.iloc[train_index],train_y.iloc[test_index]
    
    model = create_model2()
    history = model.fit(x_train,np.array(y_train).reshape(len(y_train),),batch_size = 10, epochs =100,verbose = 0,validation_data=(x_val,np.array(y_val).reshape(len(y_val),)))
    a=model.evaluate(x_val,y_val)[1]
    lst2.append(a)
    epochs = range(len(history.history['acc']))
    plt.plot(history.history["acc"],label="acc")
    plt.plot(history.history['val_acc'], label = 'val_acc')
    plt.ylabel('acc value')
    plt.xlabel('epoch')
    plt.legend(loc="upper right")
    plt.show()

# it looks like the second one wins the battle, but the validation accuracy is oscilating, we will try the final one with regularization
'''
Final model, relu, 256 hidden units, regularization and bigger learning rate 
'''

def create_model4():
    model = Sequential()
    model.add(Dense(256, input_dim=train.shape[1]-1))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    adam=Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', 
                  optimizer=adam, 
                  metrics=['acc']
                 )
    model.summary()
    return model

n_split = 3
lst4=[]# collect val accuracy 
for train_index, test_index in KFold(n_splits=n_split,shuffle = True).split(train):
    train_x=train.iloc[:,0:10]
    train_y=train['type']
    x_train,x_val=train_x.iloc[train_index,:],train_x.iloc[test_index,:]
    y_train,y_val=train_y.iloc[train_index],train_y.iloc[test_index]
    
    model = create_model4()
    history = model.fit(x_train,np.array(y_train).reshape(len(y_train),),batch_size = 10, epochs =100,verbose = 0,validation_data=(x_val,np.array(y_val).reshape(len(y_val),)))
    a=model.evaluate(x_val,y_val)[1]
    lst4.append(a)
    epochs = range(len(history.history['acc']))
    plt.plot(history.history["acc"],label="acc")
    plt.plot(history.history['val_acc'], label = 'val_acc')
    plt.ylabel('acc value')
    plt.xlabel('epoch')
    plt.legend(loc="upper right")
    plt.show()

'''
Check the result of all model and find the best model is the final model
'''
print(np.average(lst),np.average(lst1),np.average(lst2),np.average(lst4))
print(np.std(lst),np.std(lst1),np.std(lst2),np.std(lst4))
#set up callbacks 
checkpoint_name = 'Weights-{epoch:03d}--{val_acc:.5f}.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_acc', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]
# run final model 
final_model = create_model4()
history = final_model.fit(train_x,train_y,batch_size = 10, epochs =150,verbose = 0,callbacks=callbacks_list,validation_split=0.2)
test_x=test.iloc[:,:10]
test_y=test.loc[:,'type']
# load weight 0.99315 validation accuracy model at epoch 142
final_model.load_weights('Weights-142--0.99315.hdf5')
# make final classification, result is 0.98767
final_model.evaluate(test_x,test_y)
















