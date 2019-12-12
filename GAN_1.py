import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from PIL import Image
from utils_hashing import hamming_distance, get_hamm_dist_ahash, mnist_data


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


# def discriminator(x):
#     l1 = tf.nn.dropout(tf.nn.leaky_relu(tf.matmul(x,   D_W1) + D_B1, .2), .3)
#     l2 = tf.nn.dropout(tf.nn.leaky_relu(tf.matmul(l1,  D_W2) + D_B2, .2), .3)
#     l3 = tf.nn.dropout(tf.nn.leaky_relu(tf.matmul(l2,  D_W3) + D_B3, .2), .3)
#     out = tf.matmul(l3, D_W4) + D_B4
#     return out


def discriminator(image_pixel_vector, original_image_h):
    fake_image_h = average_hash_tf(image_pixel_vector)
    diff = hamm_dist(fake_image_h, original_image_h)
    return diff


def generator(z):
    l1 = tf.nn.leaky_relu(tf.matmul(z,  G_W1) + G_B1, .2)
    l2 = tf.nn.leaky_relu(tf.matmul(l1, G_W2) + G_B2, .2)
    l3 = tf.nn.leaky_relu(tf.matmul(l2, G_W3) + G_B3, .2)
    l4 = tf.nn.leaky_relu(tf.matmul(l3, G_W4) + G_B4, .2)
    out = tf.nn.tanh(tf.matmul(l4, G_W5) + G_B5)
    return out

def generator_excl_out(z):
    l1 = tf.nn.leaky_relu(tf.matmul(z, G_W1) + G_B1, .2)
    l2 = tf.nn.leaky_relu(tf.matmul(l1, G_W2) + G_B2, .2)
    l3 = tf.nn.leaky_relu(tf.matmul(l2, G_W3) + G_B3, .2)
    l4 = tf.nn.leaky_relu(tf.matmul(l3, G_W4) + G_B4, .2)
    return l4

def image_block_feature(image_pixel_vector, block_num=16):
    # block_size = int(np.sqrt(len(image_pixel_vector)/block_num))
    block_size = 7
    image_2d = image_pixel_vector.reshape([28,28])
    blocks_2d = np.ones(shape=[4,4])
    for i in range(len(blocks_2d)):
        for j in range(len(blocks_2d)):
            blocks_2d[i,j] = np.mean(image_2d[i*block_size:(i+1)*block_size,j*block_size:(j+1)*block_size])
    block_median = np.median(blocks_2d)
    blocks_2d = blocks_2d - block_median
    return blocks_2d.reshape([1, 16])

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
    original_image = data.data[2].numpy()
    original_image = np.reshape(original_image,newshape=[1, 784])
    original_image =  image_block_feature(original_image)
    # original_image = original_image - np.mean(original_image)
    y_labels_narray = np.where(original_image>0, 1, 0)
    y_labels_tensor = tf.constant(y_labels_narray, dtype=tf.float32)

    different_image = data.data[6].numpy().reshape([1,784])

    # original_image_h = average_hash_tf(original_image)
    # print(f'hash: {original_image_h}')
    # Image.fromarray(original_image.numpy()).show()

    # Initialize Graph

    # Discriminator

    # Input
    X = tf.placeholder(tf.float32, shape=[1, IMAGE_PIXELS])
    Y = tf.placeholder(tf.float32, shape=[1, 16])
    Y_2 = tf.placeholder(tf.float32, shape=[1, 784])
    # Generator

    # Input
    Z = tf.placeholder(tf.float32, shape=[1, NOISE_SIZE])

    # Layer 1 Variables
    G_W1 = tf.Variable(xavier_init([100, 256]))
    G_B1 = tf.Variable(xavier_init([256]))

    # Layer 2 Variables
    G_W2 = tf.Variable(xavier_init([256, 512]))
    G_B2 = tf.Variable(xavier_init([512]))

    # Layer 3 Variables
    G_W3 = tf.Variable(xavier_init([512, 1024]))
    G_B3 = tf.Variable(xavier_init([1024]))

    # Layer 4 Variables
    G_W4 = tf.Variable(xavier_init([1024, 784]))
    G_B4 = tf.Variable(xavier_init([784]))

    G_W5 = tf.constant(shape=[784, 16])
    G_B5 = tf.constant(shape=[16])

    # Store Variables in list
    G_var_list = [G_W1, G_B1, G_W2, G_B2, G_W3, G_B3, G_W4, G_B4, G_W5, G_B5]

    G_1 = generator(Z)
    G_2 = generator_excl_out(Z)
    # G_output = G_sample
    # D_real = discriminator(X, original_image_h)
    # D_fake = discriminator(G_sample, Y)

    G_loss_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=G_1, labels=Y))
    G_loss_2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=G_2, labels=Y_2))
    G_loss = G_loss_1 + G_loss_2
    G_opt = tf.train.AdamOptimizer(2e-4).minimize(G_loss, var_list=G_var_list)


    num_test_samples = 1
    test_noise = noise_image(num_test_samples, NOISE_SIZE)

    # Inits

    num_epochs = 2000
    n_batch = 1

    # Start interactive session
    session = tf.InteractiveSession()
    # Init Variables
    tf.global_variables_initializer().run()
    # Init Logger
    # logger = Logger(model_name='TESTGAN1', data_name='MNIST')

    g_error_store = []
    # Train

    # Iterate through epochs
    for epoch in range(num_epochs):
        # 1. Train Discriminator
        X_batch = original_image

        # 2. Train Generator
        Z_orig = noise_image(BATCH_SIZE, NOISE_SIZE)
        feed_dict = {Z: Z_orig, Y: y_labels_narray, Y_2: different_image/256}
        _, g_error = session.run([G_opt, G_loss], feed_dict=feed_dict)
        g_error_store.append(g_error)

        # display.clear_output(True)
        # Generate images from test noise_image
    test_images = session.run(generator_excl_out(Z), feed_dict={Z: Z_orig})
    pd.DataFrame(test_images.reshape([28, 28])).plot()
    plt.show()
    # test_images = vectors_to_images(test_images)
        # Log Images
        # logger.log_images(test_images, num_test_samples, epoch, n_batch, num_batches, format='NHWC')
        # Log Status
        # logger.display_status(epoch, num_epochs, n_batch, num_batches, 0, g_error)

    # pd.DataFrame(g_error_store).plot()
