import tensorflow as tf
from tensorflow import keras
import numpy

# load data
data = keras.datasets.imdb
# splitting data
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)

# encode string data into integer data for neural network
# A dictionary mapping words to an integer index
_word_index = data.get_word_index()

word_index = {k: (v+3) for k, v in _word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

# flip it around so the integers point to a word rather than the other way around
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# this function will return the decoded (human readable) reviews
def decode_review(text):
	return " ".join([reverse_word_index.get(i, "?") for i in text])

#print(decode_review(test_data[0]))

# preprocess data (using keras in-built function) (reviews are different lengths so need to make the same size)
# - if the review is greater than 250 words then trim off the extra words
# - if the review is less than 250 words add the necessary amount of 's to make it equal to 250.
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)

"""
# define model
model = keras.Sequential()
model.add(keras.layers.Embedding(88000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.summary()  # prints a summary of the model

# compile and train model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# split training data into 2 sets
# Validation data (check how well model is performing on new data, to avoid model memorising data)
# cut it down to 10000 entries
x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

# training model
fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

# testing model
results = model.evaluate(test_data, test_labels)
print(results)

# see prediction on some of the reviews
test_review = test_data[0]
predict = model.predict([test_review])
print("Review: ")
print(decode_review(test_review))
print("Prediction: " + str(predict[0]))
print("Actual: " + str(test_labels[0]))

# save model so don't need to train every time
model.save("model.h5")
"""

# load model
model = keras.models.load_model("model.h5")

# making predictions from a random review of imdb (test.txt) to see if model predictions are accurate
# transform test.txt data
def review_encode(s):
	encoded = [1]
	for word in s:
		if word.lower() in word_index:
			encoded.append(word_index[word.lower()])
		else:
			encoded.append(2)
	return encoded

# open text file, read review and use the model to predict whether it is positive or negative
with open("test.txt", encoding="utf-8") as f:
	for line in f.readlines():
		# get rid of words that will have punctuation (,.") at the end of the word
		nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"","").strip().split(" ")
		encode = review_encode(nline)
		encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250) # make the data 250 words long
		predict = model.predict(encode)
		# print(line)
		# predictions are binary so close to 1 is good review, close to 0 is bad review
		print(predict[0])
		if predict <= 0.3:
			print("bad review")
		elif 0.3 <= predict <= 0.6:
			print("medicore review")
		elif 0.6 <= predict <= 0.85:
			print("good review")
		elif predict >= 0.85:
			print("very good review")