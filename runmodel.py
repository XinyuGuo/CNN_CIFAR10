import getdata
import deep_cnn
import train
import tensorflow as tf
import time
from datetime import datetime

# Constants describing the training process.
num_epochs_per_decay = 350
moving_average_decay = 0.9999
learning_rate_decay_factor = 0.1
initial_learning_rate = 0.1

# Data directory in the local machine.
data_path = '/home/guou8j/Public_Projects/CNN_CIFAR10/cifar-10-binary/cifar-10-batches-bin'

# Sizes definition.
batch_size = 128
epoch_size = 50000

# For log
log_frequency = 10
train_dir = '/home/guou8j/Public_Projects/CNN_CIFAR10/cifar10_train_log'
max_steps = 1000000
log_device_placement = False

def runmodel():
    """Train the deep CNN model of steps"""
    with tf.Graph().as_default():
        # Global step refer to the number of batches seen by the graph.
        # Detailed explaination :https://stackoverflow.com/questions/41166681/what-does-tensorflow-global-step-mean
        global_step  = tf.contrib.framework.get_or_create_global_step()
        # Get images and labels
        # Force input pipeline to CPU:0 to avoid operations sometimes ending up on 
        # GPU and resultingin a slow down.
        with tf.device('/cpu:0'):
            eval_data = False
            images, labels = getdata.obtaindata(eval_data,data_path,batch_size)
            # print labels
            # print images
            # Build a Graph that computes the logits predictions from the deep CNN model
            logits = deep_cnn.deep_cnn(images,batch_size)

            # Caculate loss.
            loss = deep_cnn.loss(logits,labels)

            # Build a Graph that trains the model with one batch of examples and 
            # updates the model parameters.
            train_op = train.train(loss, global_step, epoch_size, batch_size, num_epochs_per_decay,
                                   initial_learning_rate, learning_rate_decay_factor, moving_average_decay)

            class _LoggerHook(tf.train.SessionRunHook):
                """Logs loss and runtime"""
                def begin(self):
                    self._step = -1
                    self._start_time = time.time()

                def before_run(self,run_context):
                    self._step += 1
                    return tf.train.SessionRunArgs(loss)

                def after_run(self, run_contex, run_values):
                    if self._step % log_frequency == 0:
                        current_time = time.time()
                        duration = current_time - self._start_time
                        self._start_time = current_time

                        loss_value = run_values.results
                        examples_per_sec = log_frequency * batch_size / duration
                        sec_per_batch = float(duration / log_frequency)

                        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                    'sec/batch)')
                        print (format_str % (datetime.now(), self._step, loss_value, examples_per_sec,
                                     sec_per_batch))

            with tf.train.MonitoredTrainingSession(
                    checkpoint_dir = train_dir,
                    hooks = [tf.train.StopAtStepHook(last_step=max_steps),
                             tf.train.NanTensorHook(loss),
                             _LoggerHook()],
                    config = tf.ConfigProto(
                        log_device_placement = log_device_placement)) as mon_sess:
                while not mon_sess.should_stop():
                    mon_sess.run(train_op)

runmodel()
