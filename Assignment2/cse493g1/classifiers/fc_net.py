from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        self.params['W1'] = np.random.normal(0, weight_scale, (input_dim, hidden_dim))
        self.params['b1'] = np.zeros(hidden_dim,)
        self.params['W2'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
        self.params['b2'] = np.zeros(num_classes,)


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # assume training pass here - affine-relu-affine-softmax

        # affine inputs are x,w,b

        W1 = self.params['W1']
        W2 = self.params['W2']

        out_1, cache_1 = affine_forward(X,W1, self.params['b1'])
        out_2, cache_2 = relu_forward(out_1)
        scores, cache_3 = affine_forward(out_2,W2,self.params['b2'])


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss, softmaxgrad = softmax_loss(scores,y)
        loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2))

        dmid,dw2,db2 = affine_backward(softmaxgrad, cache_3)
        dw2 += self.reg * W2
        drel = relu_backward(dmid, cache_2)
        dx, dw1,db1 = affine_backward(drel, cache_1)
        dw1 +=self.reg * W1

        grads = {"W1": dw1, "b1": db1, "W2": dw2, "b2": db2}


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
    



class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. For a network with L layers, the architecture will be

    {affine - relu} x (L - 1) - affine - softmax

    where the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Args:
            hidden_dims: A list of integers giving the size of each hidden layer.
            input_dim: An integer giving the size of the input.
            num_classes: An integer giving the number of classes to classify.
            reg: Scalar giving L2 regularization strength.
            weight_scale: Scalar giving the standard deviation for random
                initialization of the weights.
            dtype: A numpy datatype object; all computations will be performed using
                this datatype. float32 is faster but less accurate, so you should use
                float64 for numeric gradient checking.
        """
        
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dim_1 = input_dim
        for i in range(0,self.num_layers-1): # because self.num_layers has added the extra affine layer already above
          try:
            # 1-index the labels but the hidden_dims list is 0-indexed
            self.params['W{}'.format(i+1)] = np.random.normal(0, weight_scale, (dim_1, hidden_dims[i]))
            self.params['b{}'.format(i+1)] = np.zeros(hidden_dims[i],)
            dim_1 = hidden_dims[i]
          except:
            print(i)


        self.params['W{}'.format(self.num_layers)] = np.random.normal(0, weight_scale, (hidden_dims[len(hidden_dims)-1], num_classes))
        self.params['b{}'.format(self.num_layers)] = np.zeros(num_classes,)
        

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.
        
        Args:
            X: Array of input data of shape (N, d_1, ..., d_k)
            y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
            If y is None, then run a test-time forward pass of the model and return:
            - scores: Array of shape (N, C) giving classification scores, where
                scores[i, c] is the classification score for X[i] and class c.

            If y is not None, then run a training-time forward and backward pass and
            return a tuple of:
            - loss: Scalar value giving the loss
            - grads: Dictionary with the same keys as self.params, mapping parameter
                names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        #                                                                          #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


        out_ = X
        caches = {}
        for i in range(1,self.num_layers):
          W = self.params["W{}".format(i)]
          b = self.params["b{}".format(i)]
          out_1, cache_1 = affine_relu_forward(out_,W, b)
          out_ = out_1
          # out_, cache_ = relu_forward(out_1)
          # caches[i] = (cache_1, cache_)
          caches[i] = cache_1   

        final_W = self.params["W{}".format(self.num_layers)]
        final_b = self.params["b{}".format(self.num_layers)]
        scores, cache_final = affine_forward(out_,final_W,final_b)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss, softmaxgrad = softmax_loss(scores,y)

        d_start,dw_start,db_start = affine_backward(softmaxgrad, cache_final)
        grads[f"W{self.num_layers}"] = dw_start + (self.reg * self.params[f"W{self.num_layers}"])
        grads[f"b{self.num_layers}"] = db_start

        # print(grads.keys())

        # reg_term = 0
        dx_ = d_start
        for i in reversed(range(1,self.num_layers)):
          weight = self.params["W{}".format(i)] 
          loss += 0.5* self.reg * np.sum(weight**2) # updating loss
          dx,dw,db = affine_relu_backward(dx_,caches[i]) # upstream * cached_input
          dx_ = dx # updating upstream gradient for next round
          grads["W{}".format(i)] = dw + (self.reg * weight)
          grads["b{}".format(i)] = db

        loss += 0.5 * self.reg * np.sum(self.params["W{}".format(self.num_layers)] ** 2 ) 
        # loss += 0.5 * self.reg * reg_term
        # print(grads.keys())

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
