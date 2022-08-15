import numpy as np
import pandas as pd
import os
import cv2
from matplotlib import pyplot as plt
import math
import skimage

def rotate_on_angle(image, angle):
    # dividing height and width by 2 to get the center of the image
    height, width = image.shape[:2]
    # get the center coordinates of the image to create the 2D rotation matrix
    center = (width/2, height/2)

    # using cv2.getRotationMatrix2D() to get the rotation matrix
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)

    # rotate the image using cv2.warpAffine
    rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(width, height))

    return rotated_image

def horizontal_flip(image):
    return image[:, ::-1]

def vertically_flip(image):
    return image[::-1, :]

def gaussian_noise(image):
    pass

if __name__ == "__main__":
    img_path = "./images/img1.jpg"
    img = cv2.imread(img_path)
    img = img/255
    print(img.shape)
    var = 0.1

    rotated_img = rotate_on_angle(image = img, angle = 180)
    vertical_flipped_img = vertically_flip(img)
    horizontal_flipped_img = horizontal_flip(img)
    noisy_img = skimage.util.random_noise(img, mode='gaussian', seed = 2022, mean = 0, var = var)

    rng = np.random.default_rng(2022)
    noise = rng.normal(loc = 0, scale = var ** 0.5, size = img.shape)
    ranges = [math.floor(np.min(noise)), math.ceil(np.max(noise))]

    print(ranges, noise.shape)

    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot(1, 1, 1)
    # _ = ax1.hist(np.ravel(img), bins='auto')

    # fig2 = plt.figure()
    # ax2 = fig2.add_subplot(1, 1, 1)
    # _ = ax2.hist(np.ravel(noisy_img*255), bins='auto')

    fig = plt.figure()
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)

    _ = ax1.hist(np.ravel(img[:,:,0]), bins='auto', histtype = 'step', color = 'blue', density = True, orientation = 'vertical', align = 'left')
    _ = ax1.hist(np.ravel(img[:,:,1]), bins='auto', histtype = 'step', color = 'green', density = True, orientation = 'vertical', align = 'left')
    _ = ax1.hist(np.ravel(img[:,:,2]), bins='auto', histtype = 'step', color = 'red', density = True, orientation = 'vertical', align = 'left')
    ax1.set_xlabel('original image')
    ax1.set_ylabel('Frequency')

    _ = ax2.hist(np.ravel(noisy_img[:,:,0]), bins='auto', histtype = 'step', color = 'blue', density = True, orientation = 'vertical', align = 'left')
    _ = ax2.hist(np.ravel(noisy_img[:,:,1]), bins='auto', histtype = 'step', color = 'green', density = True, orientation = 'vertical', align = 'left')
    _ = ax2.hist(np.ravel(noisy_img[:,:,2]), bins='auto', histtype = 'step', color = 'red', density = True, orientation = 'vertical', align = 'left')
    ax2.set_xlabel('Noisy')
    ax2.set_ylabel('Frequency')

    _ = ax3.hist(np.ravel(noise[:,:,0]), bins='auto', histtype = 'step', color = 'blue', density = True, orientation = 'vertical', align = 'left')
    _ = ax3.hist(np.ravel(noise[:,:,1]), bins='auto', histtype = 'step', color = 'green', density = True, orientation = 'vertical', align = 'left')
    _ = ax3.hist(np.ravel(noise[:,:,2]), bins='auto', histtype = 'step', color = 'red', density = True, orientation = 'vertical', align = 'left')
    ax3.set_xlabel('Noisy')
    ax3.set_ylabel('Frequency')

    # plt.show()
    noise = 2*((noise - noise.min())/(noise.max() - noise.min())) - 1
    ranges = [math.floor(np.min(noise)), math.ceil(np.max(noise))]
    print(ranges, noise.shape)
    
    img_gauss = img + noise
    img_gauss = np.clip(img_gauss, 0, 1.0)

    # Generate Gaussian noise
    # gauss = np.random.normal(0,1,img.size)
    # print(gauss.max(), gauss.min())
    # print(np.unique(gauss))
    # print(gauss)
    # # gauss = gauss.astype(np.uint8)
    # gauss = (gauss - gauss.min())/(gauss.max() - gauss.min())
    # print('\n\n===========')
    # print(gauss.max(), gauss.min())
    # print(np.unique(gauss))
    # print(gauss)
    # gauss = gauss*255
    # gauss = gauss.astype(np.uint8)
    # gauss = gauss.reshape(img.shape[0],img.shape[1],img.shape[2])
    # # img_gauss = gauss + img
    # # Add the Gaussian noise to the image
    # img_gauss = cv2.add(img,gauss)

    # img = img/255
    # img_gauss = gauss + img
    # img_gauss = img_gauss.astype(np.uint8)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(3, 1, 1)
    ax2 = fig1.add_subplot(3, 1, 2)
    ax3 = fig1.add_subplot(3, 1, 3)

    _ = ax1.hist(np.ravel(img[:,:,0]), bins='auto', histtype = 'step', color = 'blue', density = True, orientation = 'vertical', align = 'left')
    _ = ax1.hist(np.ravel(img[:,:,1]), bins='auto', histtype = 'step', color = 'green', density = True, orientation = 'vertical', align = 'left')
    _ = ax1.hist(np.ravel(img[:,:,2]), bins='auto', histtype = 'step', color = 'red', density = True, orientation = 'vertical', align = 'left')
    ax1.set_xlabel('original image')
    ax1.set_ylabel('Frequency')

    _ = ax2.hist(np.ravel(img_gauss[:,:,0]), bins='auto', histtype = 'step', color = 'blue', density = True, orientation = 'vertical', align = 'left')
    _ = ax2.hist(np.ravel(img_gauss[:,:,1]), bins='auto', histtype = 'step', color = 'green', density = True, orientation = 'vertical', align = 'left')
    _ = ax2.hist(np.ravel(img_gauss[:,:,2]), bins='auto', histtype = 'step', color = 'red', density = True, orientation = 'vertical', align = 'left')
    ax2.set_xlabel('Noisy')
    ax2.set_ylabel('Frequency')

    _ = ax3.hist(np.ravel(noise[:,:,0]), bins='auto', histtype = 'step', color = 'blue', density = True, orientation = 'vertical', align = 'left')
    _ = ax3.hist(np.ravel(noise[:,:,1]), bins='auto', histtype = 'step', color = 'green', density = True, orientation = 'vertical', align = 'left')
    _ = ax3.hist(np.ravel(noise[:,:,2]), bins='auto', histtype = 'step', color = 'red', density = True, orientation = 'vertical', align = 'left')
    ax3.set_xlabel('Noisy')
    ax3.set_ylabel('Frequency')

    # plt.show()

    # plotting the images 
    # resizing before plotting
    size = (250, 250)

    img = cv2.resize(img, size)
    cv2.imshow("original_image", img)

    rotated_img = cv2.resize(rotated_img, size)
    cv2.imshow("rotated_image", rotated_img)

    vertical_flipped_img = cv2.resize(vertical_flipped_img, size)
    cv2.imshow("vertical_flipped_img", vertical_flipped_img)

    horizontal_flipped_img = cv2.resize(horizontal_flipped_img, size)
    cv2.imshow("horizontal_flipped_img", horizontal_flipped_img)

    img_gauss = cv2.resize(img_gauss, size)
    cv2.imshow("img_gauss", img_gauss)

    noisy_img = cv2.resize(noisy_img, size)
    cv2.imshow("noisy_img", noisy_img)

    plt.show()
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()