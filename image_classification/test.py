# reference from stanford CS231n "http://cs231n.github.io/classification/#intro"
from nn import NearestNeighbor 
import data_utils

Xtrain, Ytrain, Xtest, Ytest = data_utils.load_CIFAR10("cifar-10-batches-py")
# transform each image into a 1-dimensional array
Xtrain_rows = Xtrain.reshape(Xtrain.shape[0], 32 * 32 * 3)
Xtest_rows = Xtest.reshape(Xtest.shape[0], 32 * 32 * 3)

# Initialize a NearestNeighbor instance
nn = NearestNeighbor()
nn.train(Xtrain_rows, Ytrain)
# Use the classifier to predict the test examples
ypred = nn.predict(Xtest_rows)
# Calculate the accuracy and print it out
print("The accuracy is: {}".format(np.mean(ypred == Ytest)))
