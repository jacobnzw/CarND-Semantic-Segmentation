#!/usr/bin/env python3
import os.path
import tensorflow as tf
import numpy as np
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.

    Parameters
    ----------
    sess :
        TensorFlow Session

    vgg_path : str
        Path to vgg folder, containing "variables/" and "saved_model.pb"

    Returns
    -------
    : tuple
        Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)

    """

    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    in_ten = sess.graph.get_tensor_by_name(vgg_input_tensor_name)
    kp_ten = sess.graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    l3_ten = sess.graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    l4_ten = sess.graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    l7_ten = sess.graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return in_ten, kp_ten, l3_ten, l4_ten, l7_ten


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network. Build skip-layers using the vgg layers.

    Parameters
    ----------
    vgg_layer3_out :
        TF Tensor for VGG Layer 3 output

    vgg_layer4_out :
        TF Tensor for VGG Layer 4 output

    vgg_layer7_out :
        TF Tensor for VGG Layer 7 output

    num_classes :
        Number of classes to classify

    Returns
    -------
    :
        The Tensor for the last layer of output
    """

    # tmp = tf.print(vgg_layer7_out.shape.as_list())
    # last encoder layer, 1x1 convolution
    l7_out = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, strides=(1, 1), padding='same',
                              kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # decoder layers
    l4a_in = tf.layers.conv2d_transpose(l7_out, num_classes, 4, strides=(2, 2), padding='same',
                                        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    l4b_in = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='same',
                              kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    l4_out = tf.add(l4a_in, l4b_in)

    l3a_in = tf.layers.conv2d_transpose(l4_out, num_classes, 4, strides=(2, 2), padding='same',
                                        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    l3b_in = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='same',
                              kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    l3_out = tf.add(l3a_in, l3b_in)

    out = tf.layers.conv2d_transpose(l3_out, num_classes, 16, strides=(8, 8), padding='same',
                                     kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    return out


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.

    Parameters
    ----------
    nn_last_layer : TF Tensor
         The last layer in the neural network

    correct_label : TF Placeholder
         The correct label image

    learning_rate : TF Placeholder
         The learning rate

    num_classes : int
        Number of classes to classify

    Returns
    -------
    : tuple
        Tuple of (logits, train_op, cross_entropy_loss)

    """

    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1, num_classes))

    # loss function
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))

    # Adam optimizer with given learning rate
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # training operation: iteration of Adam optimizer on cross-entropy loss
    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.

    Parameters
    ----------
    sess :
        TF Session

    epochs :
        Number of epochs

    batch_size :
        Batch size

    get_batches_fn :
        Function to get batches of training data. Call using get_batches_fn(batch_size)

    train_op :
        TF Operation to train the neural network

    cross_entropy_loss :
        TF Tensor for the amount of loss

    input_image :
        TF Placeholder for input images

    correct_label :
        TF Placeholder for label images

    keep_prob :
        TF Placeholder for dropout keep probability

    learning_rate :
        TF Placeholder for learning rate

    """

    sess.run(tf.global_variables_initializer())

    print("Training...")
    print()
    for e in range(epochs):
        print("EPOCH {} ...".format(e + 1))
        for image, label in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict={input_image: image, correct_label: label, keep_prob: 0.5,
                                          learning_rate: 0.0009})
            print("Loss: = {:.3f}".format(loss))
        print()


"""
OPTIONAL: Apply the trained model to a video

OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
You'll need a GPU with at least 10 teraFLOPS to train on. https://www.cityscapes-dataset.com/

OPTIONAL: Augment Images for better results
https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network
"""


def run():
    num_classes = 2
    image_shape = (160, 576)  # KITTI dataset uses 160x576 images
    data_dir = './data'
    runs_dir = './runs'

    tests.test_for_kitti_dataset(data_dir)
    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')

        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # load layers from the VGG model
        vgg_in, vgg_kp, vgg_l3, vgg_l4, vgg_l7 = load_vgg(sess, vgg_path)
        # define the output layer of the
        out = layers(vgg_l3, vgg_l4, vgg_l7, num_classes)

        label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        # define training operation and tensors for logits and loss
        logits, train_op, ce_loss = optimize(out, label, learning_rate, num_classes)

        epochs = 50
        batch_size = 5
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, ce_loss, vgg_in, label, vgg_kp, learning_rate)

        # save inference data using
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, vgg_kp, vgg_in)


tests.test_load_vgg(load_vgg, tf)
tests.test_layers(layers)
tests.test_optimize(optimize)
tests.test_train_nn(train_nn)

if __name__ == '__main__':
    run()
