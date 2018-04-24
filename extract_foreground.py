import numpy as np
from numpy.random import normal,uniform
import random
import copy
import math
import time
import cv2
import matplotlib.pyplot as plt
from pprint import pprint

def concat_simfore_realback(sim_image,sim_segmentation,real_image):
    #compute foreground mask
    sim_mask_foreground = np.copy(sim_segmentation)
    sim_mask_foreground[sim_mask_foreground > 0] = 1
    #compute foreground part of the simulated image
    foreground_sim = np.multiply(sim_image,sim_mask_foreground)
    #compute real background image
    foreground_real = np.multiply(real_image,sim_mask_foreground)
    background_real = np.subtract(real_image,foreground_real)
    # background_real = np.multiply(real_back,sim_mask_background)
    naive_concat = np.add(foreground_sim,background_real)
    return naive_concat

def substitute_images(sim_images, sim_segmentations, real_images):
    assert len(sim_images) == len(sim_segmentations)
    assert len(sim_images) == len(real_images)
    concated_images = []
    for sim_image, sim_segmentation, real_image in zip(sim_images,sim_segmentations,real_images):
        concated_image = concat_simfore_realback(sim_image, sim_segmentation, real_image)
        concated_images.append(concated_image)
    return concated_images

def cv_read(path,channels = 3):
    assert channels in (1,3), "image should have channel size of 1 or 3!"
    if channels == 3:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = cv2.imread(path)
    return image

def read_images(size,sim_path, real_path):
    real_images = []
    sim_images = []
    sim_segmentations = []
    for i in range(size):
        #sim image
        sim_image_path = sim_path + "{0:0>6}.jpeg".format(i)
        sim_image = cv_read(sim_image_path)
        sim_images.append(sim_image)
        #sim segmentation
        sim_segmentation_path = sim_path + "{0:0>6}_segmentation.jpeg".format(i)
        sim_segmentation = cv_read(sim_segmentation_path)
        sim_segmentations.append(sim_segmentation)
        #real image
        real_image_path = real_path + "{0:0>6}.jpeg".format(i)
        real_image = cv_read(real_image_path)
        real_images.append(real_image)
    return sim_images, sim_segmentations, real_images   

def write_images2disk(images, path):
    data_size = len(images)
    for i in range(data_size):
        image = cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR)
        cv2.imwrite("sim_backSubed/{0:0>6}.jpeg".format(i),image) 
    return

def main():
    sim_path = "sim_images/"
    real_path = "real_5000/"
    subed_path = "sim_backSubed"
    #load images
    sim_images, sim_segmentations, real_images = read_images(1,sim_path, real_path)
    # plt.imshow(sim_images[0])
    plt.imshow(sim_segmentations[0])
    plt.show()
    #substitute backgrounds for sim iamges
    subed_images = substitute_images(sim_images,sim_segmentations,real_images)
    #write subed images into disk
    write_images2disk(subed_images,subed_path)

if __name__ == '__main__':
    main()
    

    