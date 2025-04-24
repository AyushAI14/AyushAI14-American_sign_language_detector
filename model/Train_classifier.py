import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


# Load the data dictionary
data_dict = pickle.load(open('./data.pickle', 'rb'))

print(data_dict.unique())

# Determine the maximum length of any sample in the data
max_length = max(len(sample) for sample in data_dict['data'])

# Pad all samples to ensure uniform length
data = np.array([np.pad(sample, (0, max_length - len(sample)), 'constant') for sample in data_dict['data']])

# Convert labels to a NumPy array
labels = np.asarray(data_dict['labels'])

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True)

# Initialize and train the model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Make predictions
y_predict = model.predict(x_test)

# Calculate accuracy
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly!'.format(score * 100))


f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()