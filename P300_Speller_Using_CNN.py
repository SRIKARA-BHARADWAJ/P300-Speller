# Importing libraries
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D

# We use sequential since we only have one input tensor and one output tensor, so functional model isnt needed
model = Sequential() 
# Consideration of input 2D matrix
# (1,64) bcz to extract feature from each electrode/channel
model.add(Conv2D(10, kernel_size=(1, 64),strides=(1,1),activation='relu',input_shape =(160, 64, 1)))
# Reduction of matrix
model.add(Conv2D(50, (13,1), activation='relu'))
# Formation of 1D array
model.add(Flatten())
# Linking and connection between layers
model.add(Dense(100, activation='relu'))
# Interconnected final layer for prediction
model.add(Dense(2, activation='softmax'))

# Training the model
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(),metrics=['accuracy'])


# Save the model
keras.models.save_model(model = model,filepath = r'S:\BE\Mini Project\Codes\Models\Base_Model.h5',overwrite = True,include_optimizer=True)

