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
    # get Hamming distance between block averaged hashes of image1 and image2
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
    # convert any array into pixel values
    side = np.sqrt(vectors.shape[1]).astype(int)
    vectors = vectors - np.min(vectors)
    vectors = vectors / np.max(vectors)
    return vectors.reshape([side, side])*256


def image_to_bool(image_pixel_vector):
    # convert image pixel vector into a boolean array of the same size
    # where True corresponds to pixel value being larger than the average
    image_pixel_vector = image_pixel_vector
    avg = np.mean(image_pixel_vector)
    image_pixel_vector_bool = image_pixel_vector > avg
    return image_pixel_vector_bool.astype(float)


def image_to_block_feature(image_pixel_vector):
    # convert full-res image into a block feature set
    image_2d = image_pixel_vector.reshape([28, 28])
    blocks_final = np.array(Image.fromarray(image_2d).convert("L").resize((8, 8), Image.ANTIALIAS))
    blocks_final = vectors_to_images(blocks_final.reshape([1, 64])).reshape([1, 64])
    return blocks_final/256


def block_feature_to_image(blocks):
    # convert feature set into a full-res image
    blocks = blocks.reshape([8, 8])
    img = np.array(Image.fromarray(blocks).resize((28, 28), Image.BICUBIC))
    img = vectors_to_images(img.reshape([1, 784]))
    return img
