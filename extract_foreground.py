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

def main():
    #load images
    sim_image = cv2.imread("sim_images/000001.jpeg")
    sim_image = cv2.cvtColor(sim_image,cv2.COLOR_BGR2RGB)
    sim_mask = cv2.imread("sim_images/000000_segmentation.jpeg")
    real_image = cv2.imread("sim_images/000000.jpeg")
    real_image = cv2.cvtColor(real_image,cv2.COLOR_BGR2RGB)

    sim_ARed = concat_simfore_realback(sim_image,sim_mask,real_image)

    plt.imshow(sim_ARed)
    plt.show()

if __name__ == '__main__':
    main()
    