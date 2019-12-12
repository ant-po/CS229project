import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from PIL import Image
from utils_hashing import hamming_distance, get_hamm_dist_ahash, mnist_data, noise_image, image_to_bool, \
    image_block_feature, vectors_to_images
from utils import Logger


IMAGE_PIXELS = 28*28
NOISE_SIZE = 100
BATCH_SIZE = 1


def xavier_init(size):
    in_dim = size[0] if len(size) == 1 else size[1]
    stddev = 1. / np.sqrt(float(in_dim))
    return tf.random_uniform(shape=size, minval=-stddev, maxval=stddev)


# def hamm_dist(tensor1, tensor2):
#     eq_boolean = tf.math.equal(tensor1, tensor2)
#     eq_numeric = tf.cast(eq_boolean, tf.float32)
#     res_bounded = tf.reduce_sum(eq_numeric)/tf.size(eq_numeric, out_type=tf.float32)
# #     res_unbounded = 1/(0.500001*-np.sign(res_bounded)+res_bounded)
#     return -tf.reshape(res_bounded, [1])


# def average_hash_tf(image_pixel_vector):
#     # takes in a vector of pixels representing an image and return a corresponding perceptual hash
#     image_pixel_vec = tf.reshape(image_pixel_vector, [1, 784])
#     avg = tf.reduce_mean(image_pixel_vec)
#     diff = tf.math.greater(image_pixel_vec, avg)
# #     img = Image.fromarray(image_pixel_vector)
# #     h = ihash.average_hash(img)
#     return diff


def real_nn(x):
    l1 = tf.nn.dropout(tf.nn.leaky_relu(tf.matmul(x,   R_W1) + R_B1, .2), .3)
    l2 = tf.nn.dropout(tf.nn.leaky_relu(tf.matmul(l1,  R_W2) + R_B2, .2), .3)
    l3 = tf.nn.dropout(tf.nn.leaky_relu(tf.matmul(l2,  R_W3) + R_B3, .2), .3)
    out = tf.nn.tanh(tf.matmul(l3, R_W4) + R_B4)
    return out


# def real_nn(image_pixel_vector, original_image_h):
#     fake_image_h = average_hash_tf(image_pixel_vector)
#     diff = hamm_dist(fake_image_h, original_image_h)
#     return diff


def hash_nn(r):
    l1 = tf.nn.leaky_relu(tf.matmul(r,  H_W1) + H_B1, .2)
    l2 = tf.nn.leaky_relu(tf.matmul(l1, H_W2) + H_B2, .2)
    l3 = tf.nn.leaky_relu(tf.matmul(l2, H_W3) + H_B3, .2)
    out = tf.nn.tanh(tf.matmul(l3, H_W4) + H_B4)
    return out


if __name__ == '__main__':

    # Load data
    data = mnist_data()
    # x=image_block_feature(noise_image(784,1))
    # Create loader with data, so that we can iterate over it
    data_loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
    # Num batches
    num_batches = len(data_loader)
    data = mnist_data()

    # Set original image
    original_image = data.data[1].numpy().reshape([1, 784])

    # Set parent image
    parent_image = data.data[2].numpy().reshape([1, 784])

    ## Initialize Graph

    # real_NN

    # Input
    X = tf.placeholder(tf.float32, shape=[1, 16])

    # Layer 1 Variables
    R_W1 = tf.Variable(xavier_init([16, 100]))
    R_B1 = tf.Variable(xavier_init([100]))
    #
    # Layer 2 Variables
    R_W2 = tf.Variable(xavier_init([100, 512]))
    R_B2 = tf.Variable(xavier_init([512]))

    # Layer 3 Variables
    R_W3 = tf.Variable(xavier_init([512, 1024]))
    R_B3 = tf.Variable(xavier_init([1024]))

    # Out Layer Variables
    R_W4 = tf.Variable(xavier_init([1024, 784]))
    R_B4 = tf.Variable(xavier_init([784]))

    # Store Variables in list
    R_var_list = [R_W1, R_B1, R_W2, R_B2, R_W3, R_B3, R_W4, R_B4]

    # Labels
    Y_R = tf.placeholder(tf.float32, shape=[1, 784])


    # Hash_NN

    # Input
    Z = tf.placeholder(tf.float32, shape=[1, 784])

    # Layer 1 Variables
    H_W1 = tf.Variable(xavier_init([784, 1024]))
    H_B1 = tf.Variable(xavier_init([1024]))

    # Layer 2 Variables
    H_W2 = tf.Variable(xavier_init([1024, 512]))
    H_B2 = tf.Variable(xavier_init([512]))

    # Layer 3 Variables
    H_W3 = tf.Variable(xavier_init([512, 128]))
    H_B3 = tf.Variable(xavier_init([128]))

    # Layer 4 Variables
    H_W4 = tf.Variable(xavier_init([128, 16]))
    H_B4 = tf.Variable(xavier_init([16]))

    # Store Variables in list
    H_var_list = [H_W1, H_B1, H_W2, H_B2, H_W3, H_B3, H_W4, H_B4]

    # Labels
    Y_H = tf.placeholder(tf.float32, shape=[1, 16])


    # Loss and Opt

    R = real_nn(X)
    H = hash_nn(R)

    R_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=R, labels=Y_R))
    R_opt = tf.train.AdamOptimizer(2e-4).minimize(R_loss, var_list=R_var_list)

    H_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=H, labels=Y_H))
    H_opt = tf.train.AdamOptimizer(2e-4).minimize(H_loss, var_list=H_var_list)

    # Training parameters

    num_epochs = 100
    n_batch = 1

    R_inputs = noise_image(1, 16)
    R_labels = image_to_bool(parent_image)
    H_labels = image_block_feature(original_image)

    # Start interactive session
    session = tf.InteractiveSession()
    # Init Variables
    tf.global_variables_initializer().run()
    # Init Logger
    logger = Logger(model_name='TESTGAN1', data_name='MNIST')

    r_loss_store = []
    h_loss_store = []
    ham_dist_store = []

    # Train
    for epoch in range(num_epochs):
        
        # 1. Train Real_NN
        feed_dict = {X: R_inputs, Y_R: R_labels}
        _, r_loss, r_output = session.run([R_opt, R_loss, R], feed_dict=feed_dict)
        r_loss_store.append(r_loss)
           
        # 2. Train Hash_NN
        feed_dict = {R: r_output, Y_H: H_labels}
        _, h_loss, h_output = session.run([H_opt, H_loss, H], feed_dict=feed_dict)
        h_loss_store.append(h_loss)
        
        # 3. Feed output of Hash_NN back into Real_NN
        R_inputs = h_output

        # Generate and log the image
        final_image = session.run(R, feed_dict={X: h_output})
        final_image = vectors_to_images(final_image)
        logger.log_images(final_image, 1, epoch, 1, 1)

        # Calculate current Hamming Distance to the original image
        ham_dist = get_hamm_dist_ahash(original_image.reshape([28, 28]), final_image)
        print(f'epoch: {epoch}, hamming distance: {ham_dist}')
        ham_dist_store.append(ham_dist)


    # Show final image
    Image.fromarray(final_image).show()

