from random import seed
from random import randint
from numpy import array
from math import ceil
from math import log10
from math import log2
from math import sqrt
from numpy import argmax
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
from keras.layers import Bidirectional

def generate_binary_string(max_len,isRand):
  bits=[]
  for i in range(0,max_len):
    if isRand==True:
      bits.append(str(randint(0,1)))
    else:
      bits.append('0')
  if isRand==False:
    bits[-1]='1'
    bits[-2]='1'
    if randint(0,1):
      bits[-1]='0'
    if randint(0,1):
      bits[-2]='0'
  s=""
  return s.join(bits)
 
def add_binary_nums(x, y):
  max_len = max(len(x), len(y))

  x = x.zfill(max_len)
  y = y.zfill(max_len)

  # initialize the result
  result = ''

  # initialize the carry
  carry = 0

  # Traverse the string
  for i in range(max_len - 1, -1, -1):
    r = carry
    r += 1 if x[i] == '1' else 0
    r += 1 if y[i] == '1' else 0
    result = ('1' if r % 2 == 1 else '0') + result
    carry = 0 if r < 2 else 1	 # Compute the carry.

  if carry !=0 : result = '1' + result

  return result.zfill(max_len)


# generate lists of random integers and their sum
def random_sum_pairs(n_examples, n_numbers, largest,isRand):
	X, y = list(), list()
	for i in range(n_examples):
		in_pattern = [generate_binary_string(ceil(log2(largest+1)),isRand) for _ in range(n_numbers)]
		out_pattern = add_binary_nums(in_pattern[0],in_pattern[1])
		X.append(in_pattern)
		y.append(out_pattern)
	return X, y
 
# convert data to strings
def to_string(X, y, n_numbers, largest):
  max_length = n_numbers * ceil(log2(largest+1)) + n_numbers - 1
  Xstr = list()
  for pattern in X:
    strp = '+'.join([str(n) for n in pattern])
    strp = ''.join([' ' for _ in range(max_length-len(strp))]) + strp
#     strp=strp[::-1]
    Xstr.append(strp)
  max_length = ceil(log2(n_numbers * (largest+1)))
  ystr = list()
  for pattern in y:
    strp = str(pattern)
    strp = ''.join([' ' for _ in range(max_length-len(strp))]) + strp
#     strp=strp[::-1]
    ystr.append(strp)
  return Xstr, ystr
 
# integer encode strings
def integer_encode(X, y, alphabet):
	char_to_int = dict((c, i) for i, c in enumerate(alphabet))
	Xenc = list()
	for pattern in X:
		integer_encoded = [char_to_int[char] for char in pattern]
		Xenc.append(integer_encoded)
	yenc = list()
	for pattern in y:
		integer_encoded = [char_to_int[char] for char in pattern]
		yenc.append(integer_encoded)
	return Xenc, yenc
 
# one hot encode
def one_hot_encode(X, y, max_int):
	Xenc = list()
	for seq in X:
		pattern = list()
		for index in seq:
			vector = [0 for _ in range(max_int)]
			vector[index] = 1
			pattern.append(vector)
		Xenc.append(pattern)
	yenc = list()
	for seq in y:
		pattern = list()
		for index in seq:
			vector = [0 for _ in range(max_int)]
			vector[index] = 1
			pattern.append(vector)
		yenc.append(pattern)
	return Xenc, yenc
 
# generate an encoded dataset
def generate_data(n_samples, n_numbers, largest, alphabet,isRand=True):
  # generate pairs
  X, y = random_sum_pairs(n_samples, n_numbers, largest,isRand)
  # convert to strings
  X, y = to_string(X, y, n_numbers, largest)
  # integer encode
  X, y = integer_encode(X, y, alphabet)
  # one hot encode
  X, y = one_hot_encode(X, y, len(alphabet))
  # return as numpy arrays
  X, y = array(X), array(y)
  return X, y
 
# invert encoding
def invert(seq, alphabet):
	int_to_char = dict((i, c) for i, c in enumerate(alphabet))
	strings = list()
	for pattern in seq:
		string = int_to_char[argmax(pattern)]
		strings.append(string)
	return ''.join(strings)
 
# define dataset
seed(1)
n_samples = 100000
n_numbers = 2
largest = 1024
alphabet = ['0', '1', '+', ' ']
n_chars = len(alphabet)
n_in_seq_length = n_numbers * ceil(log2(largest+1)) + n_numbers - 1
n_out_seq_length = ceil(log2(n_numbers * (largest+1)))
# define LSTM configuration
n_batch = 100
n_epoch = 40
# create LSTM
# model = Sequential()
# model.add(LSTM(100, input_shape=(n_in_seq_length, n_chars)))
# model.add(Bidirectional(LSTM(20, return_sequences=True), input_shape=(n_in_seq_length, n_chars), merge_mode='concat'))
# model.add(RepeatVector(n_out_seq_length))
# model.add(LSTM(50, return_sequences=False))
# model.add(TimeDistributed(Dense(n_chars, activation='softmax')))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(model.summary())

# BLSTM
print(n_in_seq_length)
print(n_chars)
model = Sequential()
model.add(LSTM(10, input_shape=(n_in_seq_length, n_chars)))
model.add(RepeatVector(n_out_seq_length))
model.add(LSTM(10, return_sequences=True))
model.add(TimeDistributed(Dense(n_chars, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# train LSTM
for i in range(n_epoch):
	X, y = generate_data(n_samples, n_numbers, largest, alphabet)
	print(i)
	model.fit(X, y, epochs=1, batch_size=n_batch)
 
# evaluate on some new patterns
X, y = generate_data(n_samples, n_numbers, largest, alphabet)
result = model.predict(X, batch_size=n_batch, verbose=0)
# calculate error
expected = [invert(x, alphabet) for x in y]
predicted = [invert(x, alphabet) for x in result]
# show some examples
for i in range(20):
  inp=""
  for j in range(0,len(X[i])):
    for k in range(0,4):
      if X[i][j][k]>0.5:
        inp+=alphabet[k]
  print(inp)
  print(expected[i])
  print(predicted[i])
  print("------------")