"""

"""
import tensorflow as tf
import urllib
import os
import sys
import tarfile
source_url= 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
EPOCH_SIZE_TRAIN=50000
EPOCH_SIZE_EVAL = 10000

# download cifar10 dataset from the data url, and unzip the ball.
def downloaddata(source_url):
    curdir = os.getcwd()
    filename = source_url.split('/')[-1]
    cifardir = filename.split('.')[0]
    filepath = os.path.join(curdir,cifardir)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    cifarpath = os.path.join(filepath,filename)
    if not os.path.exists(cifarpath):
        def _downloadstatus(count,block_size,total_size):
            sys.stdout.write('\r>>Downloading %s %.1f%%' %
                (filename,float(count*block_size)/float(total_size)*100.0))
            sys.stdout.flush()
        cifarpath, _ = urllib.urlretrieve(source_url,cifarpath,_downloadstatus)
        print()
        statinfo = os.stat(cifarpath)
        print('Sucessfully Download',filename, statinfo.st_size,'bytes.')

    cifar10_dir = os.path.join(filepath,'cifar-10-batches-bin')
    if not os.path.exists(cifar10_dir):
        tarfile.open(cifarpath,'r:gz').extractall(filepath)
    return cifar10_dir

# return cifar10 data to the application.
def obtaindata(eval_data,data_path,batch_size):
    if not eval_data:
        filenames = [os.path.join(data_path,'data_batch_%d.bin' % i)
                    for i in xrange(1,6)]
        epoch_size = EPOCH_SIZE_TRAIN
    else:
        filenames = [os.path.join(data_path,'test_batch.bin')]
        epoch_size = EPOCH_SIZE_EVAL

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # Create a queue that produces the filenames to read.
    f_q = tf.train.string_input_producer(filenames)

    # Read examples from files in the filename queue.
    file_reader = tf.FixedLengthRecordReader(record_bytes = 3073)
    key, value = file_reader.read(f_q)

    # Convert from a string to a vector of uint8 that is 3073 bytes long.
    record = tf.decode_raw(value, tf.uint8)
    # The first bytes is the label, uint8 -> int32
    label_bytes = 1
    image_bytes = 32*32*3
    label = tf.cast(tf.strided_slice(record,[0],[label_bytes]), tf.int32)

    # The remaining bytes after the label is the image. Reshape form [3*32*32]->[32*32*3]
    data_depth = tf.reshape(
                tf.strided_slice(record,[label_bytes],
                            [label_bytes + image_bytes]),[3,32,32]
            )
    # Convert from [3*32*32] -> [32*32*3]
    uint8image = tf.transpose(data_depth,[1,2,0])
    reshaped_image = tf.cast(uint8image, tf.float32)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(reshaped_image)

    # Set the shapes of tensors.
    float_image.set_shape([32,32,3])
    label.set_shape([1])

    # Set up threshold for random shuffling (capacity).
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(epoch_size*
                            min_fraction_of_examples_in_queue)
    print ('Filling queue with %d CIFAR iamges before staring to train. '
           'This will take a while.' % min_queue_examples)

    # Create bathes by randomly shuffling tensors. 
    # For this we use a queue that randomizes the order of examples,
    # Details: https://www.tensorflow.org/api_guides/python/reading_data#Reading_from_files
    images,label_batch = tf.train.shuffle_batch(
        [float_image,label],
        batch_size = batch_size,
        num_threads = 1,
        capacity = min_queue_examples + 3*batch_size,
        min_after_dequeue = min_queue_examples)

    # Display the training images in the visulizer.
    tf.summary.image('images',images)

    return  images, tf.reshape(label_batch, [batch_size])
