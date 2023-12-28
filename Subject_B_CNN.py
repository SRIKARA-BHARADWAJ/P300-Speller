
import keras
import numpy as np

#General Variables
epochs = 15
num_classes = 2
img_rows, img_cols = 160, 64

#Load Data
x_train = np.load(r'S:\BE\Mini Project\Codes\Train and Test Data\b_signal_train.npy')
x_test = np.load(r'S:\BE\Mini Project\Codes\Train and Test Data\b_signal_test.npy')
y_train = np.load(r'S:\BE\Mini Project\Codes\Train and Test Data\b_type_train.npy').astype(int)
y_test = np.load(r'S:\BE\Mini Project\Codes\Train and Test Data\b_type_test.npy').astype(int)
a_code_test = np.load(r'S:\BE\Mini Project\Codes\Train and Test Data\b_code_test.npy').astype(int)

p300 = keras.models.load_model(r'S:\BE\Mini Project\Codes\Models\Base_Model.h5')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

p300.fit(x_train, y_train, validation_split = 0.02941176, epochs = epochs, verbose = 1)


score = p300.evaluate(x_test, y_test, verbose = 1)
print("\nTest Set Validation Results-> %s: %.2f%%" % (p300.metrics_names[1], score[1]*100) ,"%s: %.2f%%" % (p300.metrics_names[0], score[0]*100) )

#save the model
keras.models.save_model(
    model = p300,
    filepath = r'S:\BE\Mini Project\Codes\Models\Subject_B_CNN.h5',
    overwrite = True,
    include_optimizer=True
)

