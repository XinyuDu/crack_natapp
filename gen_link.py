#coding:utf-8
from PIL import Image, ImageDraw, ImageFont  
import random
import numpy as np

MAX_LETTERS=6
# create a array contain letters and numbers [a b c .....z 0 1 .... 9]
charlist = [chr(i) for i in range(97,123)] + [ str(i) for i in range(10)]

CHAR_SET_LEN=len(charlist) #36

# get random letters
def getletters(): 
    num = random.sample(charlist,MAX_LETTERS)
    str=''  
    value = str.join(num)  
    return value

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
# gray pic
def convert2gray(img):
	if len(img.shape) > 2:
		gray = np.mean(img, -1)
		return gray
	else:
		return img
# generate pic with text
def gen_link_text_and_image():
	image = Image.new('RGB', (68, 24), (0,0,0))  
	font = ImageFont.truetype('./MONACO.TTF', 19)  # the font should be same as your shell
	draw = ImageDraw.Draw(image)  
	text = getletters()
	for t in range(len(text)):
		draw.text((11 * t + 1, 0), text[t], font=font, fill=(255,255,255))
	image=image.resize((160,60))
	# image.show()
	image=np.array(image)
	image=convert2gray(image)
	return text2vec(text), image

