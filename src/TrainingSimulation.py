print('Setting up')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utlis import *
from sklearn.model_selection import train_test_split

### STEP 1 Importing data
path = '../MyData'
data = importDatainfo(path)

### STEP 2 Visualization data
balanceData(data, display = False)

### STEP 3 Processing data
imagesPath, steerings = loadData(path, data)

#print(imagesPath[0], steering[0]) # check the 1st output

### STEP 4 Training and Testing Data
xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steerings, test_size=0.2, random_state=5)
print('Total Training Images: ', len(xTrain))
print('Total Validation Images: ', len(xVal))

### STEP 5 Augmenting Images

### STEP 6 Preprocessing Data

### STEP 7 Batch Generation

### STEP 8 Creating Models

model = createModel()
model.summary()

### STEP 9 Training our model

history = model.fit(batchGen(xTrain, yTrain, 10, 1),steps_per_epoch=20, epochs=5, validation_data = batchGen(xVal, yVal, 20, 0), validation_steps=20)

### STEP 10 
model.save('model.h5')
print('Model Saved')
 
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.ylim([0,1])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()





