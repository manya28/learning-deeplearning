from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]

    for i in range(num_train):
      raw_scores = np.dot(X[i], W)
      exp_scores = np.exp(raw_scores)
      loss += -np.log(exp_scores[y[i]] / np.sum(exp_scores))
      delta = exp_scores / np.sum(exp_scores)
      delta[y[i]] = delta[y[i]] - 1
      for j in range(num_classes):
        dW[:,j] += X[i] * delta[j]

    loss /= num_train
    loss += reg * np.sum(W * W)

    dW /= num_train
    dW += reg * 2* W

    # we need to find dL/dW = dL/ds * ds/dW where ds/dW = x[i]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]

    raw_scores_mat = np.dot(X,W) # 500 x 10
    exp_scores_mat = np.exp(raw_scores_mat)
    # correct_score_mat = exp_scores_mat[y]
    probs_mat = exp_scores_mat / np.sum(exp_scores_mat,axis=1).reshape(-1,1)
    loss = np.sum(-np.log(probs_mat[np.arange(probs_mat.shape[0]), y]))

    loss /= num_train
    loss += reg * np.sum(W * W)

    probs_mat[np.arange(probs_mat.shape[0]),y] -= 1

    dW = X.T.dot((probs_mat))
    # print(dW.shape)
    dW /= num_train
    dW += reg*2*W 


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
