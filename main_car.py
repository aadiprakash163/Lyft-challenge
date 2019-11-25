import os.path
import tensorflow as tf
import helper
import warnings
import random
from distutils.version import LooseVersion
import project_tests as tests
import scipy.misc
import numpy as np
from glob import glob


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
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'                               # This function gives us access to 
    vgg_layer3_out_tensor_name = 'layer3_out:0'                             # intermidiate layers of the VGG16 net    
    vgg_layer4_out_tensor_name = 'layer4_out:0'                             # on which we perform 1x1 convolution,
    vgg_layer7_out_tensor_name = 'layer7_out:0'                             # skip-layers and appropriate upsampling.

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph() # As suggested in the project walkthrough video

    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out_tensor = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out_tensor = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out_tensor = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    
    return image_input, keep_prob, layer3_out_tensor, layer4_out_tensor, layer7_out_tensor
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output


    Make ammendements in the VGG16 network to add decoder part in place of fully connected layer. And finally 
    return the final layer output tensor.

    """
    # TODO: Implement function

    conv1x1_7 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding = 'same',
                                    kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3),
                                    kernel_initializer = tf.random_normal_initializer(stddev = 0.01))

    conv1x1_4 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding = 'same',
                                    kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3),
                                    kernel_initializer = tf.random_normal_initializer(stddev = 0.01))

    conv1x1_3 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding = 'same',
                                    kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3),
                                    kernel_initializer = tf.random_normal_initializer(stddev = 0.01))

    """
    Final output architecture:

    output = [2 X {conv1x1_4 + (conv1x1_7)}X2 + conv1x1_3] X 8

    let's simplify it in the terms of variable names used

    output = [ {add1}_X 2 + conv1x1_3 ] X 8

    output = [ add2 ] X 8

    """

    conv1x1_7_X2 = tf.layers.conv2d_transpose(conv1x1_7, num_classes, 4, strides = (2,2), padding = 'same',
                                    kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3),
                                    kernel_initializer = tf.random_normal_initializer(stddev = 0.01))


    add1 = tf.add(conv1x1_4, conv1x1_7_X2)

    add1_X2 = tf.layers.conv2d_transpose(add1, num_classes, 4, strides = (2,2), padding = 'same',
                                    kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3),
                                    kernel_initializer = tf.random_normal_initializer(stddev = 0.01))

    add2 = tf.add(add1_X2, conv1x1_3)

    output = tf.layers.conv2d_transpose(add2, num_classes, 16, strides = (8,8), padding = 'same',
                                    kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3),
                                    kernel_initializer = tf.random_normal_initializer(stddev = 0.01), name = 'output')

    return output
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)


    Define the optimization problem. Basically we have to reduce the cross entropy, same as we did in term 1.
    """
    # TODO: Implement function

    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1, num_classes))

    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = correct_label))

    obj_func = tf.train.AdamOptimizer(learning_rate = learning_rate) # AdamOptimizer suggested in project walkthrough video

    train_op = obj_func.minimize(cross_entropy_loss)
    
    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
       
    """
    # TODO: Implement function
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        print("Ongoing epoch: ", epoch+1)
        for image, label in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict = {input_image: image, correct_label: label, keep_prob: 0.5, learning_rate: 0.0009}) 
            print("Loss: ", loss)
    pass
tests.test_train_nn(train_nn)



def gen_batch_function(data_folder, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """

        image_paths = glob(os.path.join(data_folder, 'CameraRGB', '*.png'))
        label_paths_road = glob(os.path.join(data_folder, 'labeled', 'car', '*.png'))
        # label_paths_car = glob(os.path.join(data_folder, 'labeled', 'car', '*.png'))
        background_color = np.array([0, 0, 0])

        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            
            # for image_file in image_paths[batch_i:batch_i+batch_size]:
            for i,image_file in enumerate(image_paths[batch_i:batch_i+batch_size]):

                image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

                # gt_image_file = label_paths[os.path.basename(image_file)]
                gt_image_file_road = label_paths_road[i]
                
                gt_image_road = scipy.misc.imresize(scipy.misc.imread(gt_image_file_road), image_shape) 

                gt_bg_road = np.all(gt_image_road == background_color, axis=2)
                gt_bg_road = gt_bg_road.reshape(*gt_bg_road.shape, 1)
                gt_image_road = np.concatenate((gt_bg_road, np.invert(gt_bg_road)), axis=2)
            
                

                images.append(image)
                gt_images.append(gt_image_road)

            yield np.array(images), np.array(gt_images)
    return get_batches_fn


def run():
    num_classes = 2
    # image_shape = (600, 800)
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    # tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    # For classifying cars
    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        


        data_folder = './CarlaData/Train/'

        #Edit this line to get carla image set
        get_batches_fn = gen_batch_function(data_folder, image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function

        epochs = 30
        batch_size = 10

        correct_label = tf.placeholder(tf.int32,[None, None, None, num_classes], name = 'correct_label')
        learning_rate = tf.placeholder(tf.float32, name = 'learning_rate')


        # TODO: Train NN using the train_nn function

        input_image, keep_prob, layer3_out_tensor, layer4_out_tensor, layer7_out_tensor = load_vgg(sess, vgg_path)

        nn_output = layers(layer3_out_tensor, layer4_out_tensor, layer7_out_tensor, num_classes)


        logits, train_op, cross_entropy_loss = optimize(nn_output, correct_label, learning_rate, num_classes)


        saver = tf.train.Saver()
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate)
        
        # Save GraphDef
        tf.train.write_graph(sess.graph_def,'.','graph_car.pb')
        # Save checkpoint
        saver.save(sess=sess, save_path="test_model_car")

        # saver.save(sess, './trainedModel_car_and_road.ckpt')

        # TODO: Save inference data using helper.save_inference_samples
        # helper.save_inference_samples(runs_dir, './test', sess, image_shape, logits, keep_prob, input_image)

        

        # OPTIONAL: Apply the trained model to a video


    


if __name__ == '__main__':
    run()
