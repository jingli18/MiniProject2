import car
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

training_images = train_data_with_label()
testing_images = test_data_with_label()

tr_img_data = np.array([i[0] for i in training_images]).reshape(-1, 64, 64, 1)
tr_lbl_data = np.array([i[1] for i in training_images])

tst_img_data = np.array([i[0] for i in testing_images]).reshape(-1, 64, 64, 1)
tst_lbl_data = np.array([i[1] for i in testing_images]) 

model = Sequential()

model.add(InputLayer(input_shape = [64, 64, 1]))
model.add(Conv2D(filters=32, kernel_size = 5, strides = 1, padding = 'same', activation = 'relu'))
model.add(MaxPool2D(pool_size = 5, padding = 'same'))

model.add(Conv2D(filters=50, kernel_size = 5, strides = 1, padding = 'same', activation = 'relu'))
model.add(MaxPool2D(pool_size = 5, padding = 'same'))

model.add(Conv2D(filters=80, kernel_size = 5, strides = 1, padding = 'same', activation = 'relu'))
model.add(MaxPool2D(pool_size = 5, padding = 'same'))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(rate = 0.5))
model.add(Dense(2, activation = 'softmax'))
optimizer = Adam(lr = 1e-3)

model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(x = tr_img_data, y = tr_lbl_data, epochs = 50, batch_size = 100)
model.summary()

fig = plt.figure(figsize = (14, 14))

for cnt, data in enumerate(testing_images[10: 10]):

	y = fig.add_subplot(6,5, cnt + 1)
	img = data[0]
	data = img.reshape(1, 64, 64, 1)
	model_out = model_predict([data])

	if np.argmax(model_out) == 1:
		str_label = 'Truck'
	else:
		str_label = 'Car'

	y.imshow(img, cmap = 'gray')
	plt.title(str_label)
	y.axes.get_xaxis().set_visible(False)
	y.axes.get_yaxis().set_visible(False) 
