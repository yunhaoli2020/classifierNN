import numpy as np
import cvxpy as cp




def SVMClassifier(train_features, train_targets, test_features, test_targets):
    """

    This function trains a one-vs-all SVM classifier using CVXPY.
    The SVM classifier predict the output value y by implementing the following formula:
    y = sign(beta.T @ x - v), where x is the input, and beta, v are the SVM parameters.
    We use CVXPY to optimize beta and v by transferring the training of SVM into a convex optimization problem.
    In order to improve the accuracy of SVM classifier, we add a l1-regularization term in the objective function.

    Input:
    train_features: features of training dataset.
    train_targets: training labels.
    test_features: features of testing dataset.
    test_labels: testing labels.

    Output:
    betaMat: trained beta vectors (for n classes, there are n corresbonding beta vectors)
    vMat: trained v values (for n classes, there are n corresponding v values)

    """
    # data pre-processing
    train_targets_svm = np.zeros((train_targets.shape[0], 10))
    test_targets_svm = np.zeros((test_targets.shape[0], 10))

    for i in range(train_features.shape[0]):
        for j in range(10):
            if train_targets[i] == j:
                train_targets_svm[i][j] = 1
            else:
                train_targets_svm[i][j] = -1

    for i in range(test_features.shape[0]):
        for j in range(10):
            if test_targets[i] == j:
                test_targets_svm[i][j] = 1
            else:
                test_targets_svm[i][j] = -1

    betaMat = np.zeros((train_features.shape[1], train_targets_svm.shape[1]))
    vMat = np.array([])
    for i in range(train_targets_svm.shape[1]):
        beta = cp.Variable((train_features.shape[1], 1))
        v = cp.Variable()
        lamda = 0.05
        obj = cp.sum(cp.pos(1 - cp.multiply(np.reshape(train_targets_svm[:,i],(train_targets_svm.shape[0], 1)), train_features @ beta - v))) + lamda * cp.norm(beta, 1)
        prob = cp.Problem(cp.Minimize(obj))
        prob.solve()
        beta1 = beta.value
        v1 = v.value   
        betaMat[:,i] = np.reshape(beta1, (beta1.shape[0], )) 
        vMat = np.append(vMat, v1)
        # test the model and calculate the accuracy using the estimated SVM parameters
        #predict = np.matmul(test_features, beta1) - v1 
        #predict = np.sign(predict)
        #error = np.abs(test_targets_svm[:,0] - np.reshape(predict, (predict.shape[0], )))
        #error = error / 2
        #accuracy = 1 - np.sum(error) / error.shape[0]
        #print("accuracy for class", i,  "is ", accuracy)
        print("training of class ", i, "is complete")


    return betaMat, vMat







