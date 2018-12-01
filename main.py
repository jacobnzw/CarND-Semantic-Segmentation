#!/usr/bin/env python3
import os.path
import tensorflow as tf
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

    # TODO Use tf.saved_model.loader.load to load the model and weights
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

    # TODO: Implement function
    return None


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.

    Parameters
    ----------
    nn_last_layer :
        TF Tensor of the last layer in the neural network

    correct_label :
        TF Placeholder for the correct label image

    learning_rate :
        TF Placeholder for the learning rate

    num_classes :
        Number of classes to classify

    Returns
    -------
    : tuple
        Tuple of (logits, train_op, cross_entropy_loss)

    """
    # TODO: Implement function
    return None, None, None


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
    # TODO: Implement function
    pass


def run():
    num_classes = 2
    image_shape = (160, 576)  # KITTI dataset uses 160x576 images
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        load_vgg(sess, vgg_path)

        # TODO: Train NN using the train_nn function

        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


tests.test_load_vgg(load_vgg, tf)
# tests.test_layers(layers)
# tests.test_optimize(optimize)
# tests.test_train_nn(train_nn)

if __name__ == '__main__':
    run()
