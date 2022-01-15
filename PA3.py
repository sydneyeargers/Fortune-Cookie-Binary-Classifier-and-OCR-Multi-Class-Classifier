"""
    title: PA3: Fortune Cookie Binary Classifier/OCR Multi-Class Classifier
    author: Sydney Yeargers
    date created: 11/7/21
    date last modified: 11/12/21
    python version: 2.7
    description: This application implements both binary and multi-class classifiers with perceptron weight updates. It
      then generates a file storing number of mistakes made and training/testing accuracies for each iteration, as well
      as the training accuracy with standard perceptron and testing accuracy with averaged perceptron for both
      classifiers.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
import regex as re
import time


def getCountVector(data, stopwords):
    """
    Converts data into a set of features, filtering out stopwords and returns a DataFrame object representing the
    feature vector.

    :param data: a DataFrame object representing a list of fortunes
    :param stopwords: a list object representing words that should not be included in feature vector
    :return: a DataFrame object representing the feature vector
    """
    vocab = []
    train_sen = []
    tokenized = []
    # for each fortune (sentence), split into lists of words and get vocabulary
    for i in range(0,len(data)):
        sen = ''
        split = data.iloc[i,0].split()  # split sentence into words
        words = [word for word in split if word not in stopwords]  # get list of words
        tokenized.append(words)
        sen += ' '.join(words)
        train_sen.append(sen)
        # don't store repeat words in vocabulary
        for word in words:
            if word not in vocab:
                vocab.append(word)
    # add and remove columns in DataFrame
    data['Filtered_Data'] = train_sen
    data['Tokenized_Data'] = tokenized
    del data['Data']
    vocab.sort()  # sort words in alphabetical order
    # create feature vector
    features = pd.DataFrame(0, data.index, vocab)
    features.reset_index(inplace=True)
    del features['index']
    index = 0
    # fill feature vector
    for j in tokenized: # for each fortune
        for n in range(0, len(j)):
            column = j[n]
            features.loc[index, column] = 1
        index += 1
    return features


def getTfidfVector(count_vector):
    """
    Finds the Term Frequency Inverse Document Frequency vector from a count vector and returns a DataFrame object
    representing the TF-IDF vector.

    :param count_vector: a DataFrame object representing a count vector (feature vector)
    :return: a DataFrame object representing a TF-IDF vector
    """
    t = TfidfTransformer()
    x = t.fit_transform(count_vector)
    tfidf_vector = pd.DataFrame(x.toarray(), index=count_vector.index, columns=count_vector.columns)
    return tfidf_vector


def getWeightBin(x, y, iter=20, lr=1):
    """
    Performs perceptron weight update for a binary classifier, and returns a Series object representing the final weight
    vector, a DataFrame object representing the final weights from every iteration, a list object representing the
    number of mistakes made every iteration, and an array object representing the average weight vector.

    :param x: a DataFrame object representing a TF-IDF vector for training data
    :param y: a list object representing the classifications of each training observation
    :param iter: an int object (default: 20) representing the number of iterations
    :param lr: an int object (default: 1) representing the learning rate
    :return: a Series object representing the final weight vector, a DataFrame object representing the final weights
        from every iteration, a list object representing the number of mistakes made every iteration, and an array
        object representing the average weight vector
    """
    weights = [0 for i in range(len(x.columns))]  # initialize weights as zero
    iter_weights = []
    mistakes = []
    # for each iteration, calculate weight vector
    for i in range(iter):
        mistake = 0
        iter_weights.append(weights)
        # for each observation, predict class
        for j in range(len(x)):
            row = x.iloc[j, ]  # a row is one observation
            prediction = binPredict(row, weights)  # predict class
            error = y[j] - prediction
            # if predicted classification is incorrect, adjust weight
            if prediction != y[j]:
                mistake += 1
                weights = weights + lr * error * row
        print('...iteration ', i + 1, '/', iter, ' complete.')
        mistakes.append(mistake)  # count mistakes for each iteration
    iter_weights = pd.DataFrame(iter_weights)  # store final weight vectors from each iteration
    avg_weight = iter_weights.mean(axis=0).values  # calculate average weight vector
    return weights, iter_weights, mistakes, avg_weight


def binPredict(row, weights):
    """
    Performs binary classification prediction on an observation using the given feature vector row and weight vector.

    :param row: a Series object representing a row of a feature vector
    :param weights: a list object representing the weight of each feature
    :return: an int object representing the predicted class for the given observation
    """
    prediction = weights[0]
    for i in range(len(row)):
        prediction += weights[i] * row[i]
    return 1 if prediction >= 0 else 0


def binPredictions(x, weight):
    """
    Performs binary classification prediction on entire dataset.

    :param x: a DataFrame object representing a feature vector
    :param weight: a list object representing the weight of each feature
    :return: a list object representing the predicted classes for each observation
    """
    predictions = []
    # for each observation, predict class
    for j in range(len(x)):
        row = x.iloc[j,]
        prediction = binPredict(row, weight)
        predictions.append(prediction)
    return predictions


def getWeightMulti(x, y, k, iter=20, lr=1):
    """
    Performs perceptron weight update for a multi-class classifier, and returns a Series object representing the final
    weight vector, a DataFrame object representing the final weights from every iteration, and a list object
    representing the number of mistakes made every iteration.

    :param x: a DataFrame object representing a feature vector for training data
    :param y: a list object representing the classifications of each training observation
    :param k: an int object representing the number of classes
    :param iter: an int object (default: 20) representing the number of iterations
    :param lr: an int object (default: 1) representing the learning rate
    :return: a Series object representing the final weight vector, a DataFrame object representing the final weights
        from every iteration, and a list object representing the number of mistakes made every iteration
    """
    classweights = []
    mistakes = []
    classes = list(set(y))  # get list of possible classes
    classes.sort()
    k = len(classes)
    # for each class, initialize weights as zero
    for k in range(k):
        classweights.append([0 for i in range(len(x.columns))])
    weights = pd.DataFrame(classweights)
    iter_weights = []
    # for each iteration, calculate weight vector
    for i in range(iter):
        predictions = []
        mistake = 0
        # for each observation, predict  class
        for rownum in range(len(x)):
            row = x.iloc[rownum].to_numpy()  # a row is one observation
            multipred = multiPredict(row, classes, weights)  # predict class
            predicted_class = multipred[0]  # get classification
            predicted_index = multipred[1]  # get index
            predictions.append(predicted_class)
            # if predicted classification is incorrect, adjust weight
            if predicted_class is not y[rownum]:
                mistake += 1
                true_class = y[rownum]
                true_index = classes.index(true_class)
                row = pd.Series(row.astype(int))
                weights.iloc[true_index] = weights.iloc[true_index].to_numpy() + (lr * row)
                weights.iloc[predicted_index] = weights.iloc[predicted_index].to_numpy() - (lr * row)
        print('...iteration ', i + 1, '/', iter, ' complete.')
        mistakes.append(mistake)  # count mistakes for each iteration
        iter_weights.append(weights.copy())  # copy weight vector for each iteration
    return weights, iter_weights, mistakes


def multiAvgWeights(iter_weights):
    """
    Finds the average weight vectors for multi-class classification.

    :param iter_weights: a DataFrame object representing weight from each iteration
    :return: a list object representing the average weight vectors for each class
    """
    avg_weights = []
    # for every class, find the average set of weight vectors
    for p in range(len(iter_weights[0].values)):
        temp = []
        avg_weight_classes = []
        # for every set of weight vectors, separate into classes
        for j in range(len(iter_weights)):
            weights = iter_weights[j]  # one set of weight vectors (all classes from one iteration)
            temp_weight = weights.iloc[p,]  # one weight vector (one class from one iteration)
            temp_vals = temp_weight.values
            temp.append(temp_vals)
        temp_avg = pd.DataFrame(temp)  # all weight vectors for one class
        avg_weight_classes = temp_avg.mean(axis=0).values  # calculate average weight vector for one class
        avg_weights.append(avg_weight_classes)  # store all average weight vectors calculated
    return avg_weights


def multiPredict(row, classes, weights):
    """
    Performs multi-class classification prediction on an observation using the given feature vector row and class
    weight vectors and returns the predicted class and class index value.

    :param row: an array object representing a row of a feature vector
    :param classes: a list object of possible classes
    :param weights: a DataFrame object representing the weight vectors for each class
    :return: a string object representing the predicted class value, and an int object representing index
    """
    predict_vector = []
    classes.sort()
    # for each class, make predictions
    for kclass in range(len(classes)):
        predict_val = []
        class_weights = weights.iloc[kclass].to_numpy()
        predict = np.dot(class_weights, row.astype(int))
        predict_val.append(predict)
        predict_val.append(kclass)
        predict_vector.append(predict_val)
    # choose most accurate prediction (max value)
    max_score = max(predict_vector, key=lambda x: x[0])
    predicted_index = max_score[1]  # get index of predicted class value
    predicted_class = classes[predicted_index]  # get predicted class
    return predicted_class, predicted_index


def multiPredictions(x, classes, weights):
    """
    Performs multi-class classification prediction on entire dataset

    :param x: a DataFrame object representing a feature vector
    :param classes: a list object of possible classes
    :param weights: a DataFrame object representing the weight vectors for each class
    :return: a list object representing the predicted classes for each observation
    """
    predictions = []
    classes.sort()
    # for each observation, predict class
    for ind, rownum in enumerate(range(len(x))):
        row = x.iloc[rownum].to_numpy()
        multipred = multiPredict(row, classes, weights)
        prediction = multipred[0]
        predictions.append(prediction)
    return predictions


def split2List(input):
    """
    Splits an input string into a list of individual numbers or letters.

    :param input: a string object representing input data
    :return: a list object representing the input broken down into individual characters or integers
    """
    return list(input)


def processFile(file):
    """
    Creates a feature vector and list of output labels from a particularly formatted csv file.

    :param file: a csv file with particular formatting
    :return: a DataFrame object representing the feature vector of predictors, and a list object representing the
    output labels
    """
    with open(file, 'r') as f:
        lines = f.readlines()
    q1 = round(len(lines)/4)  # a quarter of lines to be processed
    half = round(len(lines)/2)  # half of lines to be processed
    q3 = q1 * 3  # three quarters of lines to be processed
    i = 0
    # for each line in the file, separate input features and output labels
    for line in lines:
        split_line = re.split(r'\t+', line)  # split line on tab
        if len(split_line) > 2:
            split_dat = split2List(split_line[1][2:len(split_line[1])])  # split into individual values
            # initialize and utilize storage
            if i == 0:
                temp_x = np.array(split_dat)
                features = temp_x
                output_labels = [split_line[2]]
            # utilize storage
            else:
                temp_x = np.array(split_dat)
                features = np.append(features, temp_x)
                output_labels.append(split_line[2])
            i += 1
            # output progress to console
            if i == q1 or i == half or i == q3:
                percent = round(i/len(lines) * 100)
                print('...', percent, '% of data processed.')
    print('...100% of data processed.')
    features_dat = features.reshape(i, len(split_dat))
    input_features = pd.DataFrame.from_records(features_dat)
    return input_features, output_labels


def OCR():
    """
    Runs multi-class classification using perceptron weight update and produces desired output for program.

    :return: a string representing the output information
    """
    print('Processing training data for OCR...')
    train = processFile('OCR-data/ocr_train.txt')
    print('Processing test data for OCR...')
    test = processFile('OCR-data/ocr_test.txt')
    train_x = train[0]
    train_y = train[1]
    test_x = test[0]
    test_y = test[1]
    classes = list(set(train_y))
    print('Calculating weight vectors (multi-class)...')
    getweightmulti = getWeightMulti(train_x, train_y, 8)  # compute weight vectors for multi-class classification
    final_weight = getweightmulti[0]
    weights = getweightmulti[1]
    mistakes = getweightmulti[2]


    # a) Compute the number of mistakes made during each iteration
    print('Getting mistake counts...')
    output = ""
    # for each iteration, output number of mistakes made
    for i in range(len(mistakes)):
        output += "iteration-" + str(i+1) + " " + str(mistakes[i]) + '\n'

    # b) Compute the training accuracy and testing accuracy after each iteration
    print('Calculating training and testing accuracies...')
    # for each iteration, compute accuracies
    for kclass in range(len(weights)):
        c_weights = pd.DataFrame(weights[kclass])
        predicted_train = multiPredictions(train_x, classes, c_weights)  # predict classes for training data
        predicted_test = multiPredictions(test_x, classes, c_weights)  # predict classes for testing data
        train_accuracy = accuracy_score(train_y, predicted_train)  # compute training accuracy
        test_accuracy = accuracy_score(test_y, predicted_test)  # compute testing accuracy
        print('...iteration ', kclass+1, ' accuracies calculated.')  # update console on progress
        output += "iteration-" + str(kclass+1) + " %.3f %.3f" % (train_accuracy, test_accuracy) + '\n'

    # c) Compute the training accuracy and testing accuracy after 20 iterations
    # with standard perceptron and averaged perceptron
    print('Calculating accuracies with standard perceptron and averaged perceptron...')
    fin_tr_pred = multiPredictions(train_x, classes, final_weight)  # predict classes (standard perceptron)
    avg_weights = pd.DataFrame(multiAvgWeights(weights))  # get average weight vectors
    fin_te_pred = multiPredictions(test_x, classes, avg_weights)  # predict classes (averaged perceptron)
    stan_acc_train = accuracy_score(train_y, fin_tr_pred)  # compute training accuracy
    avg_acc_test = accuracy_score(test_y, fin_te_pred)  # compute testing accuracy
    output += "%.3f %.3f" % (stan_acc_train, avg_acc_test) + '\n'
    return output


def fortuneCookie():
    """
    Runs binary classification using perceptron weight update and produces desired output for program.

    :return: a string representing the output information
    """
    # process data
    train_data = pd.read_csv('fortune-cookie-data/traindata.txt', header=None)
    train_labels = pd.read_csv('fortune-cookie-data/trainlabels.txt', header=None)
    test_data = pd.read_csv('fortune-cookie-data/testdata.txt', header=None)
    test_labels = pd.read_csv('fortune-cookie-data/testlabels.txt', header=None)
    stop_data = pd.read_csv('fortune-cookie-data/stoplist.txt', header=None)
    range_train = train_data.size
    range_test = test_data.size
    train_data.columns = ['Data']
    train_labels.columns = ['Label']
    test_data.columns = ['Data']
    test_labels.columns = ['Label']
    stop_data.columns = ['Stop']
    # combine testing and training data
    dx = pd.concat([train_data, test_data])
    stop_words = stop_data['Stop'].tolist()
    # get feature vector
    count_vector = getCountVector(dx, stop_words)
    feat_vector = getTfidfVector(count_vector)
    train_y = train_labels['Label'].tolist()
    test_y = test_labels['Label'].tolist()
    # separate testing and training data
    train_x = feat_vector[0:range_train]
    test_x = feat_vector[range_train:range_train+range_test]

    print('Calculating weight vector (binary)...')
    getweight = getWeightBin(train_x, train_y)  # compute weight vector for binary classification
    weights = getweight[1]
    mistakes = getweight[2]
    final_weight = getweight[0]
    avg_weight = getweight[3]

    # a) Compute the number of mistakes made during each iteration
    print('Getting mistake counts...')
    output = ""
    # for each iteration, output number of mistakes made
    for i in range(len(mistakes)):
        output += "iteration-" + str(i+1) + " " + str(mistakes[i]) + '\n'

    # b) Compute the training accuracy and testing accuracy after each iteration
    print('Calculating training and testing accuracies...')
    # for each iteration, compute accuracies
    for i in range(len(weights)):
        predicted_train = binPredictions(train_x, weights.iloc[i, ])  # predict classes for training data
        predicted_test = binPredictions(test_x, weights.iloc[i, ])  # predict classes for testing data
        train_accuracy = accuracy_score(train_y, predicted_train)  # compute training accuracy
        test_accuracy = accuracy_score(test_y, predicted_test)  # compute testing accuracy
        print('...iteration ', i+1, ' accuracies calculated')  # update console on progress
        output += "iteration-" + str(i+1) + " %.3f %.3f" % (train_accuracy, test_accuracy) + '\n'

    # c) Compute the training accuracy and testing accuracy after 20 iterations
    # with standard perceptron and averaged perceptron
    print('Calculating accuracies with standard perceptron and averaged perceptron...')
    fin_tr_pred = binPredictions(train_x, final_weight)  # predict classes (standard perceptron)
    fin_te_pred = binPredictions(test_x, avg_weight)  # predict classes (averaged perceptron)
    stan_acc_train = accuracy_score(train_y, fin_tr_pred)  # compute training accuracy
    avg_acc_test = accuracy_score(test_y, fin_te_pred)  # compute testing accuracy
    output += "%.3f %.3f" % (stan_acc_train, avg_acc_test) + '\n'
    return output


def main():
    t0 = time.time()  # start timer1
    # PART ONE:
    strfc = fortuneCookie()  # perform binary classification
    timefc = time.time() - t0  # stop timer1
    print('Runtime for Fortune Cookie: ', round(timefc, 2), ' seconds.')

    t1 = time.time()  # start timer2
    # PART TWO:
    strocr = OCR()  # perform multi-class classification
    timeocr = time.time() - t1  # lap timer2
    timeall = time.time() - t0  # stop timer2
    print('Runtime for OCR: ', round(round(timeocr)/60, 2), ' minutes.')
    print('Runtime Total: ', round(round(timeall)/60, 2), ' minutes.')
    str = strfc + strocr  # create output string

    # output to .txt file
    with open('output.txt', 'w+') as output:
        output.writelines(str)


main()


