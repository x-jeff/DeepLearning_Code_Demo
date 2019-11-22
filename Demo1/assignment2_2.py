print("---------------------------------------------------")
print("1.Packages")
print("---------------------------------------------------")
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
import skimage

print("---------------------------------------------------")
print("2.Overview of the Problem set")
print("---------------------------------------------------")
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

index = 5
plt.imshow(train_set_x_orig[index])
plt.show()
print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")

m_train = np.shape(train_set_x_orig)[0]
m_test = np.shape(test_set_x_orig)[0]
num_px = np.shape(train_set_x_orig)[1]
print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))

train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.

print("---------------------------------------------------")
print("3.General Architecture of the learning algorithm")
print("---------------------------------------------------")

print("---------------------------------------------------")
print("4.Building the parts of our algorithm")
print("---------------------------------------------------")

print("---------------------------------------------------")
print("4.1.Helper functions")
print("---------------------------------------------------")
def sigmoid(z):
    s = 1 / (1+np.exp(-z))
    return s
print ("sigmoid([0, 2]) = " + str(sigmoid(np.array([0,2]))))

print("---------------------------------------------------")
print("4.2.Initializing parameters")
print("---------------------------------------------------")
def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0
    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))
    return w, b
dim = 2
w, b = initialize_with_zeros(dim)
print ("w = " + str(w))
print ("b = " + str(b))

print("---------------------------------------------------")
print("4.3.Forward and Backward propagation")
print("---------------------------------------------------")
def propagate(w, b, X, Y):
    m = X.shape[1]
    # FORWARD PROPAGATION
    A = sigmoid(np.dot(w.T,X)+b)
    cost = (-1 / m)*np.sum((np.dot(np.log(A),Y.T)+np.dot(np.log(1-A),(1-Y).T)))
    # BACKWARD PROPAGATION
    dw = (1/m) * np.dot(X,(A-Y).T)
    db = (1/m) * np.sum((A-Y))

    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    grads = {"dw": dw,
             "db": db}

    return grads, cost
w, b, X, Y = np.array([[1],[2]]), 2, np.array([[1,2],[3,4]]), np.array([[1,0]])
grads, cost = propagate(w, b, X, Y)
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print ("cost = " + str(cost))

print("---------------------------------------------------")
print("4.4.Optimization")
print("---------------------------------------------------")
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    costs = []
    for i in range(num_iterations):
        grads,cost=propagate(w,b,X,Y)
        w = w - learning_rate*grads["dw"]
        b = b - learning_rate*grads["db"]
        dw = grads["dw"]
        db = grads["db"]
        if i % 100 == 0:
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs
params, grads, costs = optimize(w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)

print ("w = " + str(params["w"]))
print ("b = " + str(params["b"]))
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print(costs)

print("---------------------------------------------------")
print("4.5.Predict")
print("---------------------------------------------------")
def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T,X)+b)
    for i in range(A.shape[1]):
        p = A[0,i]
        if p <= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1
    assert (Y_prediction.shape == (1, m))
    return Y_prediction
print ("predictions = " + str(predict(w, b, X)))

print("---------------------------------------------------")
print("5.Merge all functions into a model")
print("---------------------------------------------------")


# GRADED FUNCTION: model

def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):

    # initialize parameters with zeros
    w,b = initialize_with_zeros(X_train.shape[0])

    # Gradient descent
    params, grads, costs = optimize(w,b,X_train,Y_train,num_iterations=num_iterations,learning_rate=learning_rate,print_cost=print_cost)

    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_train = predict(w,b,X_train)
    Y_prediction_test = predict(w,b,X_test)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)

# Example of a picture that was wrongly classified.
index = 4
plt.imshow(test_set_x[:,index].reshape((num_px, num_px, 3)))
#print ("y = " + str(test_set_y[0,index]) + ", you predicted that it is a \"" + classes[int(d["Y_prediction_test"][0,index])].decode("utf-8") +  "\" picture.")
print ("y = " + str(test_set_y[0,index]) + ", you predicted that it is a \"" + classes[int(test_set_y[0,index])].decode("utf-8") +  "\" picture.")


# Plot learning curve (with costs)
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()

print("---------------------------------------------------")
print("6.Further analysis (optional/ungraded exercise)")
print("---------------------------------------------------")

learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print ("learning rate is: " + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
    print ('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()

print("---------------------------------------------------")
print("7.Test with your own image (optional/ungraded exercise)")
print("---------------------------------------------------")
my_image = "my_image2.jpg"
fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
my_predicted_image = predict(d["w"], d["b"], my_image)

plt.imshow(image)
print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")