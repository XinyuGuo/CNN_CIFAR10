import tensorflow as tf
import getdata
import deep_cnn
import numpy as np

def train(total_loss, global_step, epoch_size, batch_size, num_epochs_per_decay,
          initial_learning_rate, learning_rate_decay_factor, moving_average_decay):
    # Variable that affect learning rate.
    num_batches_per_epoch = epoch_size/batch_size
    decay_steps = int(num_batches_per_epoch*num_epochs_per_decay)

    # Decay the learning rate exponentially based on the number of steps.
    # Return a scalar tensor.
    lr = tf.train.exponential_decay(initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    learning_rate_decay_factor,
                                    staircase=True)
    tf.summary.scalar('learning_rate',lr)

    # Generate moving averages of all losses and associated summaries.
    # Compute the moving average of all individual losses and the toal loss.
    # Return an operation that updates the moving averages.
    loss_averages = tf.train.ExponentialMovingAverage(0.9,name = 'avg')
    losses = tf.get_collection('losses')
    # total_loss from deep_cnn, losses are weight decay
    loss_averages_op = loss_averages.apply(losses+[total_loss])

    # Attach a scalar summary to all individual losses and the total loss;
    # do the same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of 
        # the loss as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients
    apply_gradients_op = opt.apply_gradients(grads,global_step = global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad , var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variables_averages = tf.train.ExponentialMovingAverage(
            moving_average_decay,global_step)
    variables_averages_op = variables_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradients_op,variables_averages_op]):
        train_op = tf.no_op(name = 'train')

    return train_op
