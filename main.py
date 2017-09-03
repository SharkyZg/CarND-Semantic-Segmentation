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
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """

    
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    tf.saved_model.loader.load(sess,[vgg_tag],vgg_path)
    
    return tf.get_default_graph().get_tensor_by_name(vgg_input_tensor_name),\
           tf.get_default_graph().get_tensor_by_name(vgg_keep_prob_tensor_name),\
           tf.get_default_graph().get_tensor_by_name(vgg_layer3_out_tensor_name),\
           tf.get_default_graph().get_tensor_by_name(vgg_layer4_out_tensor_name),\
           tf.get_default_graph().get_tensor_by_name(vgg_layer7_out_tensor_name)
  
tests.test_load_vgg(load_vgg, tf)

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    conv_1x1 = tf.layers.conv2d(vgg_layer7_out, 512, 1, padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    output = tf.layers.conv2d_transpose(conv_1x1, 512, 4, strides=(2, 2), padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    conv2_1x1 = tf.layers.conv2d(vgg_layer4_out, 512, 1, padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    output_connected = tf.add(output, conv2_1x1)
    output2 = tf.layers.conv2d_transpose(output_connected, 256, 4, strides=(2, 2), padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    conv3_1x1 = tf.layers.conv2d(vgg_layer3_out, 256, 1, padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    output_connected2 = tf.add(output2, conv3_1x1)
    final_output = tf.layers.conv2d_transpose(output_connected2, num_classes, 16, strides=(8, 8), padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
 

    return final_output
tests.test_layers(layers)
  

def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    # IoU,IoUop = tf.metrics.mean_iou(tf.argmax(labels, axis = 1), tf.argmax(logits, axis = 1), num_classes)
    # total_loss = cross_entropy_loss + (1.0 - IoU)
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_operation = optimizer.minimize(cross_entropy_loss)
    return logits, training_operation, cross_entropy_loss
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
    sess.run(tf.local_variables_initializer())

    for i in range(epochs):
      batch_index = 0
      for image_batch, label_batch in get_batches_fn(batch_size): 
        batch_index += 1
        feed_dict = {input_image: image_batch, correct_label: label_batch, keep_prob: 1}
        training_loss = sess.run([train_op, cross_entropy_loss], feed_dict = feed_dict)

        print("Epoch: {} batch: {} loss: {}"
                  .format(i+1,batch_index,training_loss[1]))

tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    batch_size = 10
    num_epochs = 10
    learning_rate = 0.0001
    
    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_function = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        correct_label = tf.placeholder(tf.float32, [None, None, None, num_classes])
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        layers_output = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimize(layers_output, correct_label, learning_rate, num_classes)
        
        # TODO: Save inference data using helper.save_inference_samples
        saver = tf.train.Saver()
        saver.save(sess, './model/model.ckpt')
        print('model saved!')
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        tf.train.write_graph(sess.graph.as_graph_def(), './model', 'saved_Graph.pb',as_text=False)

        # TODO: Train NN using the train_nn function
        train_nn(sess, num_epochs, batch_size, get_batches_function, train_op, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples
        saver = tf.train.Saver()
        saver.save(sess, './model/model.ckpt')
        print('model saved!')
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        tf.train.write_graph(sess.graph.as_graph_def(), './model', 'saved_Graph.pb',as_text=False)
        
        # OPTIONAL: Apply the trained model to a video

if __name__ == '__main__':
    run()
