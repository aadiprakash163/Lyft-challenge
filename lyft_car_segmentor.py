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

def load_vgg(sess, vgg_path):
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'                               # This function gives us access to 
    vgg_layer3_out_tensor_name = 'layer3_out:0'                             # intermidiate layers of the VGG16 net    
    vgg_layer4_out_tensor_name = 'layer4_out:0'                             # on which we perform 1x1 convolution,
    vgg_layer7_out_tensor_name = 'layer7_out:0'                             # skip-layers and appropriate upsampling.
    
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out_tensor = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out_tensor = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out_tensor = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return image_input, keep_prob, layer3_out_tensor, layer4_out_tensor, layer7_out_tensor

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    
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

    add2 = tf.add(add1_X2[:,:-1,:,:], conv1x1_3)

    output = tf.layers.conv2d_transpose(add2, num_classes, 16, strides = (8,8), padding = 'same',
                                    kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3),
                                    kernel_initializer = tf.random_normal_initializer(stddev = 0.01))

    return output

def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name = 'logits')
    correct_label = tf.reshape(correct_label, (-1, num_classes))
    
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = correct_label), name = 'cross_entropy')
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate, name = 'optimizer')
    train_op = optimizer.minimize(cross_entropy_loss, name = 'train_op')
    
    return logits, train_op, cross_entropy_loss


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        print("Ongoing epoch: ", epoch+1)
        for image, label in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict = {input_image: image, correct_label: label, keep_prob: 0.5,                                       learning_rate: 0.0009}) 
            print("Loss: ", loss)
    pass
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

#                 image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                image = scipy.misc.imread(image_file)

                # gt_image_file = label_paths[os.path.basename(image_file)]
                gt_image_file_road = label_paths_road[i]
                
#                 gt_image_road = scipy.misc.imresize(scipy.misc.imread(gt_image_file_road), image_shape) 
                gt_image_road = scipy.misc.imread(gt_image_file_road) 

                gt_bg_road = np.all(gt_image_road == background_color, axis=2)
                gt_bg_road = gt_bg_road.reshape(*gt_bg_road.shape, 1)
                gt_image_road = np.concatenate((gt_bg_road, np.invert(gt_bg_road)), axis=2)
            
                

                images.append(image)
                gt_images.append(gt_image_road)

            yield np.array(images), np.array(gt_images)
    return get_batches_fn


def run():
    num_classes = 2    
    image_shape = (600, 800)
    data_dir = "./data"
    runs_dir = "./runs"
    epochs = 3
    batch_size = 15
    vgg_path = os.path.join(data_dir, 'vgg')
    data_folder = 'CarlaData/Train'
    get_batches_fn = gen_batch_function(data_folder, image_shape)
    correct_label = tf.placeholder(tf.int32, shape = [None, None, None, num_classes])
    learning_rate = tf.placeholder(tf.float32)
    
    with tf.Session() as sess:
        input_image, keep_prob, layer3_out_tensor, layer4_out_tensor, layer7_out_tensor = load_vgg(sess, vgg_path)

        nn_output = layers(layer3_out_tensor, layer4_out_tensor, layer7_out_tensor, num_classes)


        logits, train_op, cross_entropy_loss = optimize(nn_output, correct_label, learning_rate, num_classes)
    
        saver = tf.train.Saver()
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate)
        
        # Save GraphDef
        # tf.train.write_graph(sess.graph_def,'.','graph_car.pb')
        # Save checkpoint
        saver.save(sess=sess, save_path="car_segmentor")

        # saver.save(sess, './trainedModel_car_and_road.ckpt')

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, './test', sess, image_shape, logits, keep_prob, input_image)

        
if __name__ == '__main__':
    run()
    
    
    
    
    
    


       
    

    
    
    
    
