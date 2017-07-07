import math
import sys

# Store the number of features of rcv dataset.
# We can look for the maximum index when loading the dataset actually.
NUM_FEATURE_RCV = 47236 + 1
NUM_EPOC = 100
LEARNING_RATE = 0.1

def g(x):
    return 1. / (1. + math.exp(-x))

def h(params, x):
    multi = 0.
    for (idx, val) in x:
        # print (idx, val)
        multi += val * params[idx]
    return g(multi)

# Usage:
#   To read rcv train set and test set.
# Input:
#   The path to the data file.    
#   Each line in the file should be in this format:
#       `label idx1:val1 idx2:val2 idx3:val3 ...`
# Output:
#   [(label, [(idx, feature)])]
def read_rcv(rcv_path):
    samples = []
    for line in open(rcv_path):
        # label idx1:val1 idx2:val2 -> (label, idx1:val1, idx2:val2)
        split = line.split()
        # label of the sample
        label = (int(split[0]) + 1) / 2 # 1 -> 1, -1 -> 0
        # features of the sample
        features = []
        for e in split[1:]:
            # idx1:val1 -> (idx1, val1)
            e = e.split(':')
            idx, val = int(e[0]), float(e[1])
            features.append((idx, val))
        samples.append((label, features))
    print ('Finish loading ' + rcv_path)
    return samples

# Train logistic regression.
# Input: train data set
# Output: parameters
def logistic_regression_train(train_set):
    # parameters
    params = [0] * NUM_FEATURE_RCV
    for epoc in range(NUM_EPOC):
        for (label, features) in train_set:
            fac = label - h(params, features)
            # directly update the parameters with gradients
            for (idx, val) in features:
                # (fac * val) is the gradient of idx-th entry of the current sample
                params[idx] += LEARNING_RATE * fac * val
        sys.stdout.write('Finish epoc %d\r' % epoc)
        sys.stdout.flush()
    print ('Finish the training')
    return params

def logistic_regression_test(params, test_set):
    total = float(len(test_set))
    correct = 0.
    for (label, features) in test_set:
        predict = 0 if h(params, features) < 0.5 else 1
        if predict == label:
            correct += 1
    return correct / total * 100

if __name__ == '__main__':
    # load data set
    train_set = read_rcv('./rcv1_train.binary')
    test_set = read_rcv('./rcv1_test.binary')

    # train
    params = logistic_regression_train(train_set)
    # test
    accuracy = logistic_regression_test(params, test_set)
    print ("Accuracy for test set: %.2f%s" % (accuracy, "%"))
