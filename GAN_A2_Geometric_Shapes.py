# %% [code] {"execution":{"iopub.status.busy":"2021-11-15T14:26:23.31157Z","iopub.execute_input":"2021-11-15T14:26:23.311819Z","iopub.status.idle":"2021-11-15T14:26:23.327964Z","shell.execute_reply.started":"2021-11-15T14:26:23.311789Z","shell.execute_reply":"2021-11-15T14:26:23.327292Z"}}
import pandas as pd

data_frame = pd.read_csv("/Geom(1).csv")

# %% [code] {"execution":{"iopub.status.busy":"2021-11-15T14:26:23.330806Z","iopub.execute_input":"2021-11-15T14:26:23.331034Z","iopub.status.idle":"2021-11-15T14:26:23.336822Z","shell.execute_reply.started":"2021-11-15T14:26:23.331003Z","shell.execute_reply":"2021-11-15T14:26:23.335964Z"}}
opt = 'adam'
loss = 'mse'

# %% [code] {"execution":{"iopub.status.busy":"2021-11-15T14:26:23.338366Z","iopub.execute_input":"2021-11-15T14:26:23.338688Z","iopub.status.idle":"2021-11-15T14:26:23.34653Z","shell.execute_reply.started":"2021-11-15T14:26:23.33865Z","shell.execute_reply":"2021-11-15T14:26:23.34575Z"}}
X = data_frame.values 
X.shape

# %% [code] {"execution":{"iopub.status.busy":"2021-11-15T14:26:23.348515Z","iopub.execute_input":"2021-11-15T14:26:23.348954Z","iopub.status.idle":"2021-11-15T14:26:23.358215Z","shell.execute_reply.started":"2021-11-15T14:26:23.348918Z","shell.execute_reply":"2021-11-15T14:26:23.357439Z"}}
import numpy as np
from sklearn.model_selection import train_test_split

x_train, x_test = train_test_split(X,test_size=0.2, random_state=11)
x_train, x_validate = train_test_split(x_train,test_size=0.2, random_state=1)
print(x_train.shape)
print(x_test.shape)
print(x_validate.shape)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-15T14:26:23.361599Z","iopub.execute_input":"2021-11-15T14:26:23.362244Z","iopub.status.idle":"2021-11-15T14:26:23.414445Z","shell.execute_reply.started":"2021-11-15T14:26:23.362208Z","shell.execute_reply":"2021-11-15T14:26:23.413792Z"}}
import keras
from keras import layers

encoding_dim = 30

input_pts = keras.Input(shape=(122,))
a = layers.Dense(70,activation='linear')(input_pts)

encoded = layers.Dense(encoding_dim, activation='linear')(a)

b = layers.Dense(70,activation="linear")(encoded)
decoded = layers.Dense(122, activation='linear')(b)

autoencoder = keras.Model(input_pts, decoded)

encoder = keras.Model(input_pts, encoded)

encoded_input = keras.Input(shape=(encoding_dim,))
b_layer = autoencoder.layers[-2]
decoder_layer = autoencoder.layers[-1]

decoder = keras.Model(encoded_input, decoder_layer(b_layer(encoded_input)))

autoencoder.compile(optimizer=opt, loss=loss)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-15T14:26:23.415918Z","iopub.execute_input":"2021-11-15T14:26:23.416146Z","iopub.status.idle":"2021-11-15T14:26:27.213529Z","shell.execute_reply.started":"2021-11-15T14:26:23.416114Z","shell.execute_reply":"2021-11-15T14:26:27.212705Z"}}
autoencoder.fit(x_train, x_train,
                epochs=150,
                batch_size=13,
                shuffle=True,
                validation_data=(x_validate, x_validate))

# %% [code] {"execution":{"iopub.status.busy":"2021-11-15T14:26:27.215012Z","iopub.execute_input":"2021-11-15T14:26:27.215274Z","iopub.status.idle":"2021-11-15T14:26:27.364884Z","shell.execute_reply.started":"2021-11-15T14:26:27.215238Z","shell.execute_reply":"2021-11-15T14:26:27.364182Z"}}
encoded_pts = encoder.predict(x_test)
decoded_pts = decoder.predict(encoded_pts)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-15T14:26:27.366903Z","iopub.execute_input":"2021-11-15T14:26:27.367163Z","iopub.status.idle":"2021-11-15T14:26:27.374116Z","shell.execute_reply.started":"2021-11-15T14:26:27.367128Z","shell.execute_reply":"2021-11-15T14:26:27.373331Z"}}
encoded_pts.shape

# %% [code] {"execution":{"iopub.status.busy":"2021-11-15T14:26:27.375698Z","iopub.execute_input":"2021-11-15T14:26:27.376043Z","iopub.status.idle":"2021-11-15T14:26:27.382989Z","shell.execute_reply.started":"2021-11-15T14:26:27.376001Z","shell.execute_reply":"2021-11-15T14:26:27.382151Z"}}
decoded_pts.shape

# %% [code] {"execution":{"iopub.status.busy":"2021-11-15T14:26:27.384542Z","iopub.execute_input":"2021-11-15T14:26:27.384948Z","iopub.status.idle":"2021-11-15T14:26:27.394085Z","shell.execute_reply.started":"2021-11-15T14:26:27.384912Z","shell.execute_reply":"2021-11-15T14:26:27.393402Z"}}
x_test[0,0::2].shape # X coordinates

# %% [code] {"execution":{"iopub.status.busy":"2021-11-15T14:26:27.396358Z","iopub.execute_input":"2021-11-15T14:26:27.396559Z","iopub.status.idle":"2021-11-15T14:26:27.403793Z","shell.execute_reply.started":"2021-11-15T14:26:27.396536Z","shell.execute_reply":"2021-11-15T14:26:27.403073Z"}}
x_test[0,1::2].shape # Y coordinates

# %% [code] {"execution":{"iopub.status.busy":"2021-11-15T14:26:27.40492Z","iopub.execute_input":"2021-11-15T14:26:27.405307Z","iopub.status.idle":"2021-11-15T14:26:27.413917Z","shell.execute_reply.started":"2021-11-15T14:26:27.405271Z","shell.execute_reply":"2021-11-15T14:26:27.413089Z"}}
np.amax(X)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-15T14:26:27.416958Z","iopub.execute_input":"2021-11-15T14:26:27.417264Z","iopub.status.idle":"2021-11-15T14:26:29.748292Z","shell.execute_reply.started":"2021-11-15T14:26:27.41723Z","shell.execute_reply":"2021-11-15T14:26:29.747505Z"}}
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows=3,ncols=5,figsize=(50,30))

for i in range(len(x_test)):
    ax[0,i].set_xlim(xmin=-150,xmax=150)
    ax[0,i].set_ylim(ymin=-150,ymax=150)
    ax[0,i].plot(x_test[i,0::2], x_test[i,1::2],'bo')
    ax[0,i].grid(linestyle = '--', linewidth = 0.5)
    
    ax[1,i].set_xlim(xmin=-150,xmax=150)
    ax[1,i].set_ylim(ymin=-150,ymax=150)
    ax[1,i].plot(decoded_pts[i,0::2], decoded_pts[i,1::2],'ro')
    ax[1,i].grid(linestyle = '--', linewidth = 0.5)
    
    ax[2,i].set_xlim(xmin=-150,xmax=150)
    ax[2,i].set_ylim(ymin=-150,ymax=150)
    ax[2,i].plot(x_test[i,0::2], x_test[i,1::2],'bo')
    ax[2,i].set_xlim(xmin=-150,xmax=150)
    ax[2,i].set_ylim(ymin=-150,ymax=150)
    ax[2,i].plot(decoded_pts[i,0::2], decoded_pts[i,1::2],'ro')
    ax[2,i].grid(linestyle = '--', linewidth = 0.5)
    ax[2,i].set_title("MSE = "+str(round(mean_squared_error(x_test[i],decoded_pts[i]),5)),fontsize=30)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-15T14:26:29.749535Z","iopub.execute_input":"2021-11-15T14:26:29.75021Z","iopub.status.idle":"2021-11-15T14:26:29.756433Z","shell.execute_reply.started":"2021-11-15T14:26:29.75018Z","shell.execute_reply":"2021-11-15T14:26:29.755705Z"}}
mean_squared_error(x_test,decoded_pts)

# %% [markdown]
# # 60

# %% [code] {"execution":{"iopub.status.busy":"2021-11-15T14:26:29.757926Z","iopub.execute_input":"2021-11-15T14:26:29.758505Z","iopub.status.idle":"2021-11-15T14:26:29.809818Z","shell.execute_reply.started":"2021-11-15T14:26:29.758462Z","shell.execute_reply":"2021-11-15T14:26:29.80904Z"}}
# This is the size of our encoded_2 representations
encoding_dim = 60  

# This is our input image
input_pts = keras.Input(shape=(122,))
# "encoded_2" is the encoded_2 representation of the input
a_2 = layers.Dense(70,activation='linear')(input_pts)
encoded_2 = layers.Dense(encoding_dim, activation='linear')(a_2)
# "decoded_2" is the lossy reconstruction of the input

b_2 = layers.Dense(70,activation='linear')(encoded_2)
decoded_2 = layers.Dense(122, activation='linear')(b_2)

# This model maps an input to its reconstruction
autoencoder = keras.Model(input_pts, decoded_2)

# This model maps an input to its encoded_2 representation
encoder = keras.Model(input_pts, encoded_2)

# This is our encoded_2 (32-dimensional) input

encoded_2_input = keras.Input(shape=(encoding_dim,))

# Retrieve the last layer of the autoencoder model

b_layer_2 = autoencoder.layers[-2]

decoder_layer = autoencoder.layers[-1]

# Create the decoder model
decoder = keras.Model(encoded_2_input, decoder_layer(b_layer_2(encoded_2_input)))

autoencoder.compile(optimizer=opt, loss=loss)
# autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

print(" ")

# %% [code] {"execution":{"iopub.status.busy":"2021-11-15T14:26:29.811191Z","iopub.execute_input":"2021-11-15T14:26:29.811439Z","iopub.status.idle":"2021-11-15T14:26:33.287436Z","shell.execute_reply.started":"2021-11-15T14:26:29.811406Z","shell.execute_reply":"2021-11-15T14:26:33.286651Z"}}
autoencoder.fit(x_train, x_train,
                epochs=150,
                batch_size=13,
                shuffle=True,
                validation_data=(x_validate, x_validate))

# %% [code] {"execution":{"iopub.status.busy":"2021-11-15T14:26:33.28862Z","iopub.execute_input":"2021-11-15T14:26:33.288904Z","iopub.status.idle":"2021-11-15T14:26:33.439737Z","shell.execute_reply.started":"2021-11-15T14:26:33.288865Z","shell.execute_reply":"2021-11-15T14:26:33.439058Z"}}
encoded_2_pts = encoder.predict(x_test)
decoded_2_pts = decoder.predict(encoded_2_pts)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-15T14:26:33.442713Z","iopub.execute_input":"2021-11-15T14:26:33.443174Z","iopub.status.idle":"2021-11-15T14:26:33.449944Z","shell.execute_reply.started":"2021-11-15T14:26:33.443144Z","shell.execute_reply":"2021-11-15T14:26:33.449129Z"}}
encoded_2_pts.shape

# %% [code] {"execution":{"iopub.status.busy":"2021-11-15T14:26:33.451177Z","iopub.execute_input":"2021-11-15T14:26:33.452032Z","iopub.status.idle":"2021-11-15T14:26:33.457245Z","shell.execute_reply.started":"2021-11-15T14:26:33.451987Z","shell.execute_reply":"2021-11-15T14:26:33.456382Z"}}
decoded_pts.shape

# %% [code] {"execution":{"iopub.status.busy":"2021-11-15T14:26:33.45836Z","iopub.execute_input":"2021-11-15T14:26:33.458715Z","iopub.status.idle":"2021-11-15T14:26:33.466739Z","shell.execute_reply.started":"2021-11-15T14:26:33.458679Z","shell.execute_reply":"2021-11-15T14:26:33.465902Z"}}
x_test[0,0::2].shape

# %% [code] {"execution":{"iopub.status.busy":"2021-11-15T14:26:33.468164Z","iopub.execute_input":"2021-11-15T14:26:33.468861Z","iopub.status.idle":"2021-11-15T14:26:33.474778Z","shell.execute_reply.started":"2021-11-15T14:26:33.468812Z","shell.execute_reply":"2021-11-15T14:26:33.474002Z"}}
x_test[0,1::2].shape

# %% [code] {"execution":{"iopub.status.busy":"2021-11-15T14:26:33.476137Z","iopub.execute_input":"2021-11-15T14:26:33.476507Z","iopub.status.idle":"2021-11-15T14:26:35.552355Z","shell.execute_reply.started":"2021-11-15T14:26:33.476416Z","shell.execute_reply":"2021-11-15T14:26:35.551654Z"}}
fig, ax = plt.subplots(nrows=3,ncols=5,figsize=(50,30))

for i in range(len(x_test)):
    ax[0,i].set_xlim(xmin=-150,xmax=150)
    ax[0,i].set_ylim(ymin=-150,ymax=150)
    ax[0,i].plot(x_test[i,0::2], x_test[i,1::2],'bo')
    ax[0,i].grid(linestyle = '--', linewidth = 0.5)
    
    ax[1,i].set_xlim(xmin=-150,xmax=150)
    ax[1,i].set_ylim(ymin=-150,ymax=150)
    ax[1,i].plot(decoded_2_pts[i,0::2], decoded_2_pts[i,1::2],'ro')
    ax[1,i].grid(linestyle = '--', linewidth = 0.5)
    
    ax[2,i].set_xlim(xmin=-150,xmax=150)
    ax[2,i].set_ylim(ymin=-150,ymax=150)
    ax[2,i].plot(x_test[i,0::2], x_test[i,1::2],'bo')
    ax[2,i].grid(linestyle = '--', linewidth = 0.5)
    ax[2,i].set_xlim(xmin=-150,xmax=150)
    ax[2,i].set_ylim(ymin=-150,ymax=150)
    ax[2,i].plot(decoded_2_pts[i,0::2], decoded_2_pts[i,1::2],'ro')
    ax[2,i].grid(linestyle = '--', linewidth = 0.5)
    ax[2,i].set_title("MSE = "+str(round(mean_squared_error(x_test[i],decoded_2_pts[i]),5)),fontsize=30)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-15T14:26:35.553764Z","iopub.execute_input":"2021-11-15T14:26:35.554193Z","iopub.status.idle":"2021-11-15T14:26:35.560336Z","shell.execute_reply.started":"2021-11-15T14:26:35.554158Z","shell.execute_reply":"2021-11-15T14:26:35.559702Z"}}
mean_squared_error(x_test,decoded_2_pts)
