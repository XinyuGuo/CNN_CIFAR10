# Deep Convolutional Neuralnetwork for Image Recogonition 

Assignment 2 @ Standford U Online Course CS231n Convolutional Neural Networks for Visual Recogonition
Course website : http://cs231n.github.io/

In this assignment, a 11-layer deep cnn was built for CIFAR10 image recogonition with TensorFlow.
CIFAR10 data website:https://www.cs.toronto.edu/~kriz/cifar.html 

The architecture of the deep deep cnn:

cov-> ReLu-> max pooling-> local response normalization-> cov-> ReLu-> local response normalization-> 
max pooling-> dense layer(ReLu)-> dense layer(ReLu)-> softmax layer

**Highlights of the Project**

(1).Important building blocks including convolution, rectified linear activation, max pooling, and local
    response normalization.

(2).Add a learning rate schedule that systematically decreases the learning rate over time.

(3).Routines of calculating the moving average of learning parameters and using the averages during the evaluation
    to boost the predictive performance.

(4).Visulization with TensorBoard: losses and distributions of activations and gradients. 

(5).Prefetching queues for input data to isolate the model from disk latency and expensive image pre-processin.
