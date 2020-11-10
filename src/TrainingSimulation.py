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

### STEP 5 Augmenting Data

