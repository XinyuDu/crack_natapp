import time
import numpy as np
from gen_link import gen_link_text_and_image
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K
from keras.models import model_from_json  
from keras.utils.vis_utils import plot_model
K.set_image_dim_ordering('tf')

charlist = [chr(i) for i in range(97,123)] + [ str(i) for i in range(10)]

IMAGE_HEIGHT=60
IMAGE_WIDTH=160
MAX_LETTERS=6
CHAR_SET_LEN=len(charlist)

# convert text to vector
def text2vec(text):
	vector = np.zeros(MAX_LETTERS*CHAR_SET_LEN)
	for i in range(MAX_LETTERS):
		k = charlist.index(text[i])
		vector[k+i*36]=1
	return vector
# convert vector to text
def vec2text(vec):
	char_pos = vec.nonzero()[0]
	if char_pos.shape[0]!=6:
		return '######'

	text=[]
	for i in range(MAX_LETTERS):
		char_index = char_pos[i]-36*i
		if char_index >36:
			return '******'
		else:
			char = charlist[char_index]
			text.append(char)

	return "".join(text)

# generate data
def get_next_batch(batch_size=128):
	batch_x = np.zeros([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH])
	batch_y = np.zeros([batch_size, MAX_LETTERS*CHAR_SET_LEN])

	for i in range(batch_size):
		text, image = gen_link_text_and_image()
		batch_x[i,:] = image
		batch_y[i,:] = text
	return batch_x, batch_y

# binaryzation output vector
def binoutput(vec):
	bvec = np.zeros(MAX_LETTERS*CHAR_SET_LEN)
	for m in range(MAX_LETTERS):
		temp = vec[m*36:(m+1)*36]
		maxindex = np.where(temp==temp.max())[0][0]
		bvec[maxindex+m*36] = 1
	return bvec

# load model
json_file = open('my_model_architecture.json', 'r')
loaded_model_json = json_file.read()
json_file.close()  
model = model_from_json(loaded_model_json)
model.load_weights('my_model_weights.h5')

#output cnn structure pic
plot_model(model, to_file='model1.png',show_shapes=True)

#start test
j=0
n=1000
for i in range(n):
	(x,t) = get_next_batch(1)
	x = x.reshape(x.shape[0], IMAGE_HEIGHT, IMAGE_WIDTH, 1).astype('float32')
	x = x / 255
	#print (t)
	result = model.predict(x)
	#result = result[0]
	result = binoutput(result[0])

	char_t = vec2text(t[0])
	char_result = vec2text(result)

	if char_t==char_result:
		print(i, char_t,char_result) 
		j=j+1
	else:
		print('{},{},{}--wrong!'.format(i,char_t,char_result))

print('准确率:{}'.format(j/n))

