from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *



class FullyConnectedNetBasic(object):
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
          # grads["gamma{}".format(i)] = dgamma
          # grads["beta{}".format(i)] = dbeta


        loss += 0.5 * self.reg * np.sum(self.params["W{}".format(self.num_layers)] ** 2 ) 

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads





class FullyConnectedNetImproved(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
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
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        
        dim_1 = input_dim
        for i in range(0,self.num_layers-1): # because self.num_layers has added the extra affine layer already above
            # 1-index the labels but the hidden_dims list is 0-indexed
            self.params['W{}'.format(i+1)] = np.random.normal(0, weight_scale, (dim_1, hidden_dims[i]))
            self.params['b{}'.format(i+1)] = np.zeros(hidden_dims[i],)
            if self.normalization == "batchnorm":
              self.params['gamma{}'.format(i+1)] = np.ones(hidden_dims[i],)
              self.params['beta{}'.format(i+1)] = np.zeros(hidden_dims[i],)
            dim_1 = hidden_dims[i]
          # except:
          #   print(i)


        self.params['W{}'.format(self.num_layers)] = np.random.normal(0, weight_scale, (hidden_dims[len(hidden_dims)-1], num_classes))
        self.params['b{}'.format(self.num_layers)] = np.zeros(num_classes,)


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.
        
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
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out_ = X
        caches = {}
        for i in range(1,self.num_layers):
          W = self.params["W{}".format(i)]
          b = self.params["b{}".format(i)]
          if self.normalization == "batchnorm":
            gamma = self.params["gamma{}".format(i)]
            beta = self.params["beta{}".format(i)]
            bn_param = self.bn_params[i-1]
            out_1, cache_1 = affine_bn_relu_forward(out_,W, b,gamma,beta,bn_param)
          else:
            out_1, cache_1 = affine_relu_forward(out_,W,b)
          
          # print(out_1)
          if self.use_dropout:
            out_,c = dropout_forward(out_1, self.dropout_param)
            caches[i] = [cache_1, c]
            # print(len(caches))
          else:
            out_ = out_1
            caches[i] = [cache_1]

        final_W = self.params["W{}".format(self.num_layers)]
        final_b = self.params["b{}".format(self.num_layers)]
        scores, cache_final = affine_forward(out_,final_W,final_b)


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the   #
        # scale and shift parameters.                                              #
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
          if self.use_dropout:
            dx_ = dropout_backward(dx_,caches[i][1])
          weight = self.params["W{}".format(i)] 
          loss += 0.5* self.reg * np.sum(weight**2) # updating loss
          if self.normalization == "batchnorm":
            dx,dw,db,dgamma,dbeta = affine_bn_relu_backward(dx_,caches[i][0]) # upstream * cached_input
            grads["gamma{}".format(i)] = dgamma
            grads["beta{}".format(i)] = dbeta
          else:
            dx,dw,db = affine_relu_backward(dx_, caches[i][0])
          dx_ = dx # updating upstream gradient for next round
          grads["W{}".format(i)] = dw + (self.reg * weight)
          grads["b{}".format(i)] = db


        loss += 0.5 * self.reg * np.sum(self.params["W{}".format(self.num_layers)] ** 2 )

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
    
