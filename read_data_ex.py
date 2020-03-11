import numpy as np
import csv

filename = 'data//online_shoppers_intention.csv'

n = 12330
d = 18
data = []
features = []
with open(filename, newline='') as file:
	reader = csv.reader(file, delimiter = ',')
	for i, row in enumerate(reader):
		if i == 0:
			features = row
		else:
			data.append(row)

print('Features: \n', features, '\n')
print('Data: \n', data[0:5]) # first five rows of our data

# at this point, our data includes strings and numbers (still scored as strings)
# we need to convert everything to numeric values, then put in an np.array
# for example, change 'FALSE' -> 0, 'TRUE' -> 1, and use some sort of encoding for 'Visitor_Type'

# the last row of our data is binary REVENUE: let's try to predict this
X = np.zeros((n,d-1))
Y = np.zeros(n) 
for i in range(n):
	if data[i][-1] == 'TRUE': Y[i] = 1
	elif data[i][-1] == 'FALSE': Y[i] = 0
	for j in range(d-1):
		if j != 10 and j != 15:
			try:
				X[i,j] = float(data[i][j])
			except:
				if data[i][j] == 'TRUE': X[i,j] = 1
				elif data[i][j] == 'FALSE': X[i,j] = 0

# TODO: should probably rescale/normalize each feature to be on the same scale

# separate into training and testing data
num_train = int(.8 * n)
num_test = n - num_train
X_train = X[0:num_train, :]
X_test = X[num_train:,:]
Y_train = Y[0:num_train]
Y_test = Y[num_train:]

# now we're ready to train models on this data

# extra comment