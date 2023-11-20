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

    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    # Calculate the number of samples and classes
    num_samples = X.shape[0]
    num_classes = W.shape[1]

    # Loop over each sample
    for i in range(num_samples):
        
        # Calculate scores for each class 
        Xi_scores = X[i].dot(W)

        # Shift scores to avoid numerical instability due to large values
        Xi_scores -= np.max(Xi_scores)

        # Softmax probabilities for each class
        softmax_probs = np.exp(Xi_scores) / np.sum(np.exp(Xi_scores))

        # Cross-entropy loss for probabilities
        loss += -np.log(softmax_probs[y[i]])

        # Calculate the gradient 
        for j in range(num_classes):

            # For correct class
            if j == y[i]:
                dW[:, j] += (softmax_probs[j] - 1) * X[i]

            # For incorrect class
            else:
                dW[:, j] += softmax_probs[j] * X[i]

    # Average the loss over all data points
    loss /= num_samples

    # Average the gradient over all data points
    dW /= num_samples

    # Add regularization to the loss
    loss += 0.5 * reg * np.sum(W * W)

    # Adding regularization to the gradient
    dW += reg * W

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

    num_samples = X.shape[0]

    # Compute scores for all samples at once
    Xi_scores = X.dot(W)

    # Shift scores to avoid numerical instability due to large values
    Xi_scores -= np.max(Xi_scores, axis=1, keepdims=True)

    # Calculate softmax probabilities for all samples
    softmax_probs = np.exp(Xi_scores) / np.sum(np.exp(Xi_scores), axis=1, keepdims=True)

    # Correct class probabilities
    correct_class_probs = softmax_probs[np.arange(num_samples), y]

    # Cross-entropy loss
    loss = -np.sum(np.log(correct_class_probs))

    # Adjusting probabilities for correct classes
    softmax_probs[np.arange(num_samples), y] -= 1

    # Calculating the gradient 
    dW = X.T.dot(softmax_probs)

    # Average the loss over all samples
    loss /= num_samples

    # Average the gradient over all samples
    dW /= num_samples

    # Add regularization term to the loss
    loss += 0.5 * reg * np.sum(W * W)

    # Adding regularization to the gradient
    dW += reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
