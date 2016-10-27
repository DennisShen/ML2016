# @ Author : Dennis Shen
# @ Date   : 2016.10.18
# @ Title  : Spam Email Classification

from __future__ import division
import pandas as pd
import numpy  as np
import matplotlib
matplotlib.use('Agg')   # noqa
import matplotlib.pyplot as plt
import sys

""" Sigmoid Function """
def sigmoid(x):
  return 1 / (1 + np.exp(-x))
  
""" Cross Entropy """
def croEntropy(f_x, y):
    return -(y*np.log(f_x) + (1.0-y)*np.log(1.0-f_x))

""" Load data into a list """
def load_data(filename):
    df = pd.read_csv(filename, header=None)
    
    # Drop id column
    df.drop(0, axis=1, inplace=True)
    print '\nData preview:'
    print df.head()
    
    data = df.values
    return data
  
""" Generate dataset for training and validation """
def generate_dataset(data, sample=3000):
    data_X = []
    data_y = []

    total_data_num = data.shape[0]
    
    # Sample data if needed
    if sample == 'all':
        index = list(range(total_data_num))
    else:
        index = np.random.choice(total_data_num, sample, replace=False)

    # Insert data into list
    for idx in index:
        data_X.append(data[idx][:57])
        data_y.append(data[idx][57])
        
    return data_X, data_y
    
""" Add bias in the end of training data """
def make_data(train_X, train_y):
    X = np.array(train_X)
    y = np.array(train_y)
    
    # Add bias
    X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
    
    return X, y
    
""" split validation set """
def split_validation(X, y):
    # random pick 1/4 data as validation set
    random_indexes = np.random.permutation(len(y))
    train_inds = random_indexes[:int((0.75*len(y)))]
    valid_inds = random_indexes[int((0.75*len(y))):]
    return X[train_inds], y[train_inds], X[valid_inds], y[valid_inds]

""" calculate the accuracy of current weight """
def calcAccuracy(X, W, y):
    pred = np.array(sigmoid(X.dot(W)))
    pred[pred>0.5]  = 1
    pred[pred<0.5]  = 0

    err = 0
    for idx in range(len(y)):
        err = err + 1 if pred[idx] != y[idx] else err
    return 1 - err/len(y)
    
""" logistic regression """
def logReg(X, y, X_test=None, y_test=None, lr=1e-7, batch=1, lamb=0,
           epoch=10000, print_every=100):
    # Initialize weight
    W = np.random.randn(X.shape[1]) / X.shape[1] / X.shape[0]
    L_train = []
    A_train = []
    L_test  = []
    A_test  = []
    
    # AdaGrad
    G = np.zeros(W.shape)
    
    for i in range(epoch):
        # batch
        b = 0
        idx = []
        
        for j in np.random.permutation(X.shape[0]):
            idx.append(j)
            b += 1
            
            if b >= batch:
                # prediction error
                pred_err = sigmoid(X[idx].dot(W)) - y[idx]
        
                # calculate gradient
                grad_X = X[idx].T.dot(pred_err)
                grad_regular = lamb * W * batch / X.shape[0]
                grad = grad_X + grad_regular
                
                # calculate weight
                G += grad**2
                W -= grad*lr/np.sqrt(G+1e-8)
                
                # reset parameter
                b = 0
                idx = []

        # prevent inf and nan
        cross_Entropy = croEntropy(sigmoid(X.dot(W)), y)
        cross_Entropy[np.isinf(cross_Entropy)] = 1000
        cross_Entropy[np.isnan(cross_Entropy)] = 0
        
        # calculate loss and accuracy   
        Loss_train = cross_Entropy.sum() + lamb*(W**2).sum()
        Accuracy_train = calcAccuracy(X, W, y)
        
        L_train.append(Loss_train)
        A_train.append(Accuracy_train)
        
        # testing
        if X_test is not None and y_test is not None:
            cross_Entropy = croEntropy(sigmoid(X_test.dot(W)), y_test)
            cross_Entropy[np.isinf(cross_Entropy)] = 1000
            cross_Entropy[np.isnan(cross_Entropy)] = 0
            
            Loss_test = cross_Entropy.sum() + lamb*(W**2).sum()
            Accuracy_test = calcAccuracy(X_test, W, y_test)
            
            L_test.append(Loss_test)
            A_test.append(Accuracy_test)
        
        # print out the progress
        if i % print_every == 0:
            if X_test is not None and y_test is not None:
                print('\tepoch: %d; loss: %.4f; Acc_train: %.4f; Acc_test: %.4f' %
                      (i, Loss_train, Accuracy_train, Accuracy_test))
            else:
                print('\tepoch: %d; loss: %.4f; Acc_train: %.4f' %
                      (i, Loss_train, Accuracy_train))
                
    print('\nfinal loss: %.4f' % L_train[-1])
    print('final training accuracy: %.4f' % A_train[-1])
    print('final testing accuracy: %.4f' % A_test[-1])

    return W, L_train, A_train, L_test, A_test

"""" generative model """
def probGen(X, y):    
    data_class1 = [x for idx, x in enumerate(X) if y[idx]==1]
    data_class2 = [x for idx, x in enumerate(X) if y[idx]==0]

    # transform to numpy array
    data_class1_array = np.matrix(data_class1)
    data_class2_array = np.matrix(data_class2)
    
    # delete answer column
    data_class1_array = np.delete(data_class1_array, 57, 1)
    data_class2_array = np.delete(data_class2_array, 57, 1)
    
    # calculate number
    N1 = data_class1_array.shape[0]
    N2 = data_class2_array.shape[0]
    N  = N1 + N2

    # calculate mean
    u1 = np.mean(data_class1_array, axis=0)
    u2 = np.mean(data_class2_array, axis=0)

    # minus mean
    data_class1_array_zero_mean = data_class1_array - u1
    data_class2_array_zero_mean = data_class2_array - u2
    
    # square
    data_class1_square = data_class1_array_zero_mean.T*(data_class1_array_zero_mean)
    data_class2_square = data_class2_array_zero_mean.T*(data_class2_array_zero_mean)
    
    # variance
    var1 = data_class1_square/N1
    var2 = data_class2_square/N2
    var  = var1*N1/N + var2*N2/N
    
    # avoid singular
    var_mod = var + np.eye(var.shape[0])*1e-6
    var_inv = np.linalg.inv(var_mod)
    
    # weight
    W = (u1 - u2)*(var_inv)
    b = 0.5*u2*var_inv*(u2.T) - 0.5*u1*var_inv*(u1.T) + np.log(N1/N2)
    W = np.concatenate((W, b), axis=1)
    
    # accuracy
    Accuracy = calcAccuracy(X, W.T, y)
    
    return W.T, Accuracy
    
""" Spam classification main program """
def run_spam_classify(trainFile, model, split=True, sample = 'all', methods = 'logistic_regression'):
    # load training data in to a list
    data = load_data(trainFile)
    
    # Picking out training data
    train_X, train_y = generate_dataset(data, sample)
    
    # Training data formatting
    X, y = make_data(train_X, train_y)
    
    # Split validation set
    if split:
        X_train, y_train, X_valid, y_valid = split_validation(X, y)
    else:
        X_train, y_train = X, y
        X_valid = None
        y_valid = None
        
    # Train data
    print '\nstart training...'
    
    if methods == 'logistic_regression':
        print '\nlogistic regression...\n'
        result = logReg(X_train, y_train, X_valid, y_valid, lr=1e-2, batch=len(y_train), 
                        lamb=0, epoch=5000, print_every=1000)
        W, L_train, A_train, L_test, A_test = result
        
        # draw the loss and Accuracy
        plt.figure(figsize=(8, 6))
        plt.title('Loss')
        plt.plot(L_train, label='train Loss', color='b')
        plt.plot(L_test,  label='valid Loss', color='r')
        plt.legend()
        plt.savefig('Loss.png', dpi=200)

        plt.figure(figsize=(8, 6))
        plt.title('Accuracy')
        plt.plot(A_train, label='train Accuracy', color='b')
        plt.plot(A_test, label='valid Accuracy', color='r')
        plt.legend()
        plt.savefig('Accuracy.png', dpi=200)

        np.savetxt(model, W)
    elif methods == 'probabilistic_generative':
        print '\nprobabilistic generative...\n'
        W, Accuracy = probGen(X_train, y_train)
        
        print('Accuracy: %.4f' % Accuracy)
        np.savetxt(model, W)
        
""" Make Kaggle Classification """
def make_test_classify(testFile, Wfile, outFile):
    # load weight
    W = np.loadtxt(Wfile)
    
    # load test data
    test = load_data(testFile)
    
    test_X = []
    for idx in range(test.shape[0]):
        test_X.append(test[idx])
        
    # Add bias
    X = np.array(test_X)
    X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)

    # pred
    pred = np.array(sigmoid(X.dot(W)))
    pred[pred>0.5] = 1
    pred[pred<0.5] = 0

    # output file
    with open(outFile, 'w') as f:
        f.write('id,label\n')
        for idx in range(600):
            f.write('%d,%d\n'%(idx+1, pred[idx]))
    
if __name__ == '__main__':
    if(sys.argv[1] == 'train'):
        if(sys.argv[2] == 'logistic_regression'):
            run_spam_classify(sys.argv[3], sys.argv[4], split=True,  sample='all', methods='logistic_regression')
        elif(sys.argv[2] == 'probabilistic_generative'):
            run_spam_classify(sys.argv[3], sys.argv[4], split=False, sample='all', methods='probablistic_generative')
    elif(sys.argv[1] == 'test'):
        make_test_classify(sys.argv[3], sys.argv[2], sys.argv[4])
