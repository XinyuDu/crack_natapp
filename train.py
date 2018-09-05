import time
import numpy as np
from gen_link import gen_link_text_and_image
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.models import model_from_json 
K.set_image_dim_ordering('tf')

time_start=time.time()
seed = 7
np.random.seed(seed)

charlist = [chr(i) for i in range(97,123)] + [ str(i) for i in range(10)]

IMAGE_HEIGHT=60
IMAGE_WIDTH=160
MAX_LETTERS=6
CHAR_SET_LEN=len(charlist)

# generate training and validation data
def get_next_batch(batch_size=128):
	batch_x = np.zeros([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH])
	batch_y = np.zeros([batch_size, MAX_LETTERS*CHAR_SET_LEN])

	for i in range(batch_size):
		text, image = gen_link_text_and_image()
		batch_x[i,:] = image
		batch_y[i,:] = text

	return batch_x, batch_y

def cnn_model():
	# create model
	num_classes = CHAR_SET_LEN*MAX_LETTERS
	model = Sequential()
	#con layer 1
	model.add(Conv2D(32, (5, 5), strides=(2, 2), padding='same', input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2),padding='same'))
	model.add(Dropout(0.2))
	#con layer 2
	model.add(Conv2D(64, (5, 5), padding='same',activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	#con layer 3
	model.add(Conv2D(64, (5, 5), padding='same', activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2),padding='same'))
	model.add(Dropout(0.2))
	#flatten for next layer
	model.add(Flatten())
	#full connection layer
	model.add(Dense(2160, activation='relu'))
	#out layer
	model.add(Dense(num_classes, activation='sigmoid'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# build the model
model = cnn_model()

(X_train, y_train) = get_next_batch(60000)
X_train = X_train.reshape(X_train.shape[0], IMAGE_HEIGHT, IMAGE_WIDTH, 1).astype('float32')
X_train = X_train / 255

(X_test, y_test) = get_next_batch(10000)
X_test = X_test.reshape(X_test.shape[0], IMAGE_HEIGHT, IMAGE_WIDTH, 1).astype('float32')
X_test = X_test / 255

# load model
# json_file = open('my_model_architecture.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()  
# model = model_from_json(loaded_model_json)
# model.load_weights('my_model_weights.h5')
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# scores = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=400, verbose=2)

while True:
	scores = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=1000, verbose=2)
	vacc = scores.history['val_acc'][-1]
	acc = scores.history['acc'][-1]
	if acc >= 0.995 and vacc >=0.999:
		break

#save model and weight
print('start to save model to files')
json_string = model.to_json()
open('my_model_architecture.json','w').write(json_string)
model.save_weights('my_model_weights.h5')
print('model saved')

time_end=time.time()
print('totally cost',time_end-time_start,'s')