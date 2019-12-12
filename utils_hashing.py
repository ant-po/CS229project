import numpy as np
import imagehash as ih

from torchvision import transforms, datasets
from PIL import Image

def mnist_data():
    compose = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    out_dir = '{}/dataset'.format('./tf_data/VGAN/MNIST')
    return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)


def hamming_distance(hash_vector1, hash_vector2, normalised=False):
    hamm_dist = hash_vector1 - hash_vector2
    if normalised:
        hamm_dist = hamm_dist/len(hash_vector1)
    return hamm_dist


def get_hamm_dist_ahash(image1, image2):
    # data = mnist_data()
    image1 = Image.fromarray(image1)
    image2 = Image.fromarray(image2)
    image1_hash = ih.average_hash(image1)
    image2_hash = ih.average_hash(image2)
    return hamming_distance(image1_hash, image2_hash)


def noise_image(n_rows, n_cols):
    noise_img = np.random.normal(size=(n_rows, n_cols))
    noise_img = noise_img - np.min(noise_img)
    noise_img = noise_img/np.max(noise_img)
    return noise_img


def images_to_vectors(images):
    return images.reshape([784])


def vectors_to_images(vectors):
    vectors = vectors - np.min(vectors)
    vectors = vectors / np.max(vectors)
    return vectors.reshape([28, 28])*256


def image_to_bool(image_pixel_vector):
    image_pixel_vector = image_pixel_vector/256
    avg = np.mean(image_pixel_vector)
    image_pixel_vector_bool = image_pixel_vector > avg
    return image_pixel_vector_bool.astype(float)


def image_block_feature(image_pixel_vector, block_num=16):
    block_size = 7
    image_2d = image_pixel_vector.reshape([28, 28])
    blocks_2d = np.ones(shape=[4, 4])
    for i in range(len(blocks_2d)):
        for j in range(len(blocks_2d)):
            blocks_2d[i, j] = np.mean(image_2d[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size])
    blocks_final = image_to_bool(blocks_2d.reshape([1, 16]))
    return blocks_final