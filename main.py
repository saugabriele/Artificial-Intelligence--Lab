import numpy as np
import sklearn
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate

#CREATING THE TRAINING TEST AND TEST_SET
#In this case to create the second training set I have used the following features:
#     number of black pixel in the image
#     number of black pixels in the middle two columns
#     number of black pixels in the middle two rows

X_train_list = []
y_train_list = []
X_train_personal_features = []

training_data_file = open("C:/Users/Utente/PycharmProjects/Artificial_Intelligence/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

for record in training_data_list:
    all_values = record.split(',')

    # scale and shift the inputs
    inputs = [int(x) for x in (np.asfarray(all_values[1:]) / 128)]
    X_train_list.append(inputs)

    #creating the training set with the three features selected before
    black_pixels = inputs.count(1)/784
    reshaped = (np.array(inputs)).reshape((28, 28))
    black_pixels_rows = (np.count_nonzero(reshaped[13] == 1) + np.count_nonzero(reshaped[14] == 1))/56
    reshaped = reshaped.T
    black_pixels_cols = (np.count_nonzero(reshaped[13] == 1) + np.count_nonzero(reshaped[14] == 1))/56
    X_train_personal_features.append([black_pixels, black_pixels_rows, black_pixels_cols])

    # all_values[0] is the target label for this record
    y_train_list.append(int(all_values[0]))

    pass

X_train = np.array(X_train_list)
X_train_features = np.array(X_train_personal_features)
y_train = np.array(y_train_list)

test_data_file = open("C:/Users/Utente/PycharmProjects/Artificial_Intelligence/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

X_test_list = []
X_test_personal_features = []
y_test_list = []

for record_test in test_data_list:
        all_values_test = record_test.split(',')

        #correct answer is first value
        correct_label = int(all_values_test[0])
        y_test_list.append(correct_label)

        #scale and shift the inputs
        inputs_test = [int(x) for x in (np.asfarray(all_values_test[1:]) /128)]
        X_test_list.append(inputs_test)

        # creating the test set with the three features selected before
        black_pixels = inputs_test.count(1)/784
        reshaped_test = (np.array(inputs_test)).reshape((28, 28))
        black_pixels_rows = (np.count_nonzero(reshaped_test[13] == 1) + np.count_nonzero(reshaped_test[14] == 1))/56
        reshaped = reshaped_test.T
        black_pixels_cols = (np.count_nonzero(reshaped[13] == 1) + np.count_nonzero(reshaped[14] == 1))/56
        X_test_personal_features.append([black_pixels, black_pixels_rows, black_pixels_cols])

        pass

X_test = np.array(X_test_list)
X_test_features = np.array(X_test_personal_features)
y_test = np.array(y_test_list)

#CREATING A GRAPH TO EVALUATE THE DISCRIMINATING ABILITY OF THE THREE FEATURES (ONLY 1000 PATTERNS)
n_classes = len(np.unique(y_train))
colours = ['blue','red','yellow','green','purple','brown','gray','cyan','olive','orange']
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

for k in range(n_classes):
    X = X_train_features[y_train == k, 0]
    Y = X_train_features[y_train == k, 0]
    Z = X_train_features[y_train == k, 0]
    ax.scatter(Z, X, Y, c= colours[k], label = 'class' + str(k))

plt.legend()
plt.show()

#CROSS VALIDATION FOR EVALUATE DIFFERENT ARTIFICIAL NEURAL NETWORK STRUCTURES
#First model with a lower number of hidden units
model_1 = MLPClassifier(hidden_layer_sizes=[15], activation='logistic', max_iter=1500, tol=0.0001)

cv_result = cross_validate(model_1, X_train_features, y_train, cv=5)
mean_score = cv_result['test_score'].mean()

print("\nMLP classifier accuracy on training set with three features using a lower number of hidden units in the"
      " hidden layer: ", mean_score)

#Second model with a higher number of hidden units
model_2 = MLPClassifier(hidden_layer_sizes=[30], activation='logistic', max_iter=1500, tol=0.0001)

cv_result = cross_validate(model_2, X_train_features, y_train, cv=5)
mean_score = cv_result['test_score'].mean()

print("\nMLP classifier accuracy on training set with three features using a high number of hidden units in the "
      "hidden layer: ", mean_score)

#Third model with two hidden layers
model_3 = MLPClassifier(hidden_layer_sizes=[30, 20], activation='logistic', max_iter=1000, tol=0.0001)

cv_result = cross_validate(model_3, X_train_features, y_train, cv=5)
mean_score = cv_result['test_score'].mean()

print("\nMLP classifier accuracy on training set with three features using two hidden layers: ", mean_score)

#Fourth model with three hidden layers
model_4 = MLPClassifier(hidden_layer_sizes=[40, 30, 20], activation='logistic', max_iter=500, tol=0.0001)

cv_result = cross_validate(model_4, X_train_features, y_train, cv = 5)
mean_score = cv_result['test_score'].mean()

print("\nMLP classifier accuracy on training set with three features using three hidden layers: ", mean_score)

#First model with a lower number of hidden units using all the pixels as features
model_1 = MLPClassifier(hidden_layer_sizes=[50], activation='logistic', max_iter=500, tol=0.0001)

cv_result = cross_validate(model_1, X_train, y_train, cv=5)
mean_score = cv_result['test_score'].mean()

print("\nMLP classifier accuracy on training set using a lower number of hidden units in the hidden layer: ", mean_score)

#Second model with a higher number of hidden units using all the pixels as features
model_2 = MLPClassifier(hidden_layer_sizes=[100], activation='logistic', max_iter=500, tol=0.0001)

cv_result = cross_validate(model_2, X_train, y_train, cv=5)
mean_score = cv_result['test_score'].mean()

print("\nMLP classifier accuracy on training set using a high number of hidden units in the hidden layer: ", mean_score)

#Third model with two hidden layers using all the pixels as features
model_3 = MLPClassifier(hidden_layer_sizes=[100, 50], activation='logistic', max_iter=500, tol=0.0001)

cv_result = cross_validate(model_3, X_train, y_train, cv=5)
mean_score = cv_result['test_score'].mean()

print("\nMLP classifier accuracy on training set using two hidden layers: ", mean_score)

#Fourth model with three hidden layers using all the pixels as features
model_4 = MLPClassifier(hidden_layer_sizes=[100, 50, 25], activation='logistic', max_iter=500, tol=0.0001)

cv_result = cross_validate(model_4, X_train, y_train, cv = 5)
mean_score = cv_result['test_score'].mean()

print("\nMLP classifier accuracy on training set using three hidden layers: ", mean_score)


#REALIZING THE ARTIFICIAL NEURAL NETWORK AND CHECKING ITS PERFORMANCE ON THE TEST SET

model = MLPClassifier(hidden_layer_sizes=[40, 30, 20], activation='logistic', max_iter=800, tol=0.0001)
model_pixels = MLPClassifier(hidden_layer_sizes=[100, 50], activation='logistic', max_iter=500, tol=0.0001)

model.fit(X_train_features, y_train)
model_pixels.fit(X_train, y_train)

p_test = model.predict(X_test_features)
p_test_pixels = model_pixels.predict(X_test)

acc_test = accuracy_score(y_test, p_test)
acc_test_pixels = accuracy_score(y_test, p_test_pixels)


print("Accuracy test with only three features: ", acc_test)
print("Accuracy test with one feature for each pixel: ", acc_test_pixels)