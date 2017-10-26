import tensorflow as tf
"""
image dimension: 3*32*32*Batch_Size
input dimension: number of images * 32 * 32 * 3
filter size : 5*5*3
filter number : 64
pooling window size : 2*2
weight decay parameter: 0.04
Note: only dense connectioned part has weight decay.
"""
def deep_cnn(image,batch_size):
    """
    conv1
    """
    with tf.variable_scope('conv1') as scope:
        # define the filter set for the first convolutional layer.
        conv_filter = tf.get_variable('W',[5,5,3,64],tf.float32,
                tf.truncated_normal_initializer(stddev=5e-2,dtype=tf.float32))

        # get the first convolutional layer
        # strides = [1,1,1,1], zero padding scheme is applied.
        conv1 = tf.nn.conv2d(image,conv_filter,[1,1,1,1],padding = 'SAME',name ='conv1')

        # define biases for each filter.
        biases =tf.get_variable("b",[64],tf.float32,initializer=tf.constant_initializer(0.0))

        # add bias to each feature map.
        # Note: the second parameter is a 1D tensor with size matching the last dimension of the
        # first parameter.
        weight_sum = tf.nn.bias_add(conv1,biases)

        # calculate the activation.
        activation = tf.nn.relu(weight_sum, name=scope.name)

    # pool1
    pool1 = tf.nn.max_pool(activation,ksize=[1,2,2,1],strides=[1,2,2,1],
            padding='SAME',name='pool1')

    # norm1
    norm1 = tf.nn.lrn(pool1,4,bias=1.0,alpha=0.001/9.0,beta=0.75,name='norm1')

    """
    conv2
    """
    with tf.variable_scope('conv2') as scope:
        # define the filter set for the second convolutional layer.
        conv_filter = tf.get_variable('W',[5,5,64,64],tf.float32,
                tf.truncated_normal_initializer(stddev=5e-2,dtype=tf.float32))

        # get the second convolutional layer
        # strides = [1,1,1,1], zero padding scheme is applied.
        conv2 = tf.nn.conv2d(norm1,conv_filter,[1,1,1,1],padding = 'SAME', name = 'conv2')

        # define biases for each filter.
        biases = tf.get_variable("b",[64],tf.float32,initializer=tf.constant_initializer(0.1))

        # add bias to each feature map.
        weight_sum = tf.nn.bias_add(conv2,biases)

        # calculate the activation.
        activation = tf.nn.relu(weight_sum, name=scope.name)

    # norm2
    norm2 = tf.nn.lrn(conv2,4,bias=1.0,alpha=0.001/9.0,beta=0.75,name='norm2')

    # pool2
    pool2 = tf.nn.max_pool(norm2,ksize=[1,2,2,1],strides=[1,2,2,1],
            padding='SAME',name='pool2')

    # dense1 
    with tf.variable_scope('dense1') as scope:
        reshape_data = tf.reshape(pool2,[batch_size,-1])
        input_dim = reshape_data.get_shape()[1].value
        first_layer_weights = tf.get_variable('W',[input_dim,384],tf.float32,
                tf.truncated_normal_initializer(stddev=0.04,dtype=tf.float32))

        # add weight decay term
        weight_decay =  tf.multiply(tf.nn.l2_loss(first_layer_weights), 0.04, name="weight_decay")
        tf.add_to_collection('losses',weight_decay)

        # biases
        biases = tf.get_variable("b",[384],tf.float32,tf.constant_initializer(0.1))

        # activation  
        dense1 = tf.nn.relu(tf.matmul(reshape_data,first_layer_weights)+biases, name=scope.name)

    # dense2
    with tf.variable_scope('dense2') as scope:
        second_layer_weights = tf.get_variable("W",[384,192],tf.float32,
                tf.truncated_normal_initializer(stddev=0.04,dtype=tf.float32))

        # add weight decay term
        weight_decay =  tf.multiply(tf.nn.l2_loss(second_layer_weights), 0.04, name="weight_decay")
        tf.add_to_collection('losses',weight_decay)

        # biases
        biases = tf.get_variable("b",[192],tf.float32,tf.constant_initializer(0.1))

        # activation  
        dense2 = tf.nn.relu(tf.matmul(dense1,second_layer_weights)+biases, name=scope.name)

    # softmax linear layer
    with tf.variable_scope('softmax_linear') as scope:
        softmax_weights = tf.get_variable("W",[192,10],tf.float32,
                tf.truncated_normal_initializer(stddev=1/192.0,dtype=tf.float32))

        biases = tf.get_variable("b",[10],tf.float32,tf.constant_initializer(0.0))
        softmax = tf.add(tf.matmul(dense2,softmax_weights),biases,name=scope.name)

    return softmax

def loss(logits, labels):
    # Calculate the average cross entropy loss across the batch.
    labels =  tf.cast(labels,tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels = labels, logits = logits, name = 'cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name = 'cross_entropy')
    tf.add_to_collection('losses',cross_entropy_mean)
    # The total loss is defined as the cross entropy loss plus all of the weight decay terms (L2 loss)
    return tf.add_n(tf.get_collection('losses'),name = 'total_loss')
