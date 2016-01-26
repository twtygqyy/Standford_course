import numpy as np
from random import shuffle

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
  num_train = X.shape[0]
  num_classes = W.shape[1]
  
  for i in xrange(num_train):
    scores = X[i].dot(W)
    scores = np.exp(scores) / sum(np.exp(scores))

    for j in xrange(num_classes):
      if j == y[i]:
        J = -np.log(scores[y[i]])
        dW[:,j] -= X[i] * (1-scores[y[i]])
      else:
        J = 0
        dW[:,j] -= X[i] * (0-scores[j])
      loss += J

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss /= num_train
  dW /= num_train
  
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = np.exp(X.dot(W))
  sum = np.sum(scores,axis = 1)
  sum_matrix = np.tile(sum, (num_classes,1)).transpose()
  scores = scores/sum_matrix
  
  values = scores[np.arange(len(scores)), y]
  loss = np.sum(-np.log(values))  
  loss = loss/num_train   
  loss += 0.5 * reg * np.sum(W * W)   
  
  g = np.zeros(scores.shape)
  g = scores
  g[np.arange(len(scores)), y] = g[np.arange(len(scores)), y] - 1
  dW = X.transpose().dot(g)   
  dW /= num_train  
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

