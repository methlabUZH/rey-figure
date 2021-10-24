import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
from skimage.io import imsave
from cv2 import imread
import os
from config import DATA_DIR
import shutil


import preprocess
from dataloader import DATA


def visualize_labels(dataset):
    n, bins, patches = plt.hist(x=dataset.labels, rwidth=0.85, bins=36)
    plt.xlabel('Label')
    plt.ylabel('Frequency')
    plt.title('Histogram of labels of {} images of the data_preprocessing set {}'.format(dataset.labels.shape, dataset.name))
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(top=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.show()

def visualize_mean_image(dataset):
    mean_image = np.mean(dataset.images, axis=0)
    plt.title('Mean image of the data_preprocessing set {}'.format(dataset.name))
    plt.imshow(mean_image, cmap='gray')
    plt.show()

def save_image(img, filename, border=False):
    if border:
        img = np.pad(img,((5,5),(5,5)),'constant')
    imsave("../data_preprocessing/visualized/preprocessing/"+filename,img)

def visualize_preprocessing(dataset, i):
    # visualize preprocessing steps of the i-th image in the dataset
    image = imread(dataset.files[i].path)
    img_gray = rgb2gray(image)
    save_image(img_gray, '0_grayscale.jpg', border=True)
    img_gray = preprocess.gamma_correct(img_gray)
    save_image(img_gray, '0.5_gamma.jpg', border=True)
    img_gray = preprocess.procedure_min(img_gray)
    save_image(img_gray, '1_iterative_minimum_filter.jpg', border=True)
    normalized = preprocess.normalization(img_gray)
    thresh_cut = np.percentile(img_gray, 4)
    img_gray = preprocess.cutdown(img_gray, thresh_cut)
    save_image(img_gray, '2_cropped.jpg', border=True)
    fill_cval = 1  # color of the padded area, probably this line should be adapted if normalized
    resized_img = preprocess.resize_padded(img_gray, preprocess.CANVAS_SIZE, fill_cval=fill_cval)
    save_image(resized_img, '3_resized.jpg', border=True)

def show_pixel_histogram(img, name):
    n, bins, patches = plt.hist(x=img.ravel(), rwidth=0.85, bins=128, range=(0, 1), log=False)
    plt.xlabel('Pixel intensity')
    plt.ylabel('Count')
    plt.title('Log histogram of pixel intensities '+name)
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(top=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.show()

def show_average_pixel_histogram(dataset):
    mean_image = np.mean(dataset.images, axis=0)
    show_pixel_histogram(mean_image, "mean image")

def save_pixel_intensities_to_files(dataset, n = 50, directory = DATA_DIR + "visualized/"):
    # create directory, delete all files in it
    if not os.path.exists(directory):
        os.mkdir(directory)
    dir = directory + "pixel_intensities/"
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.mkdir(dir)

    # go through n items in dataset and write comparison to disk
    for ind, image in enumerate(dataset.images):
        if ind % 10:
            print("Writing file {} from {}".format(ind,len(dataset.images)))
        fig = plt.figure(figsize=(15,5))
        ax = fig.add_subplot(1,2,1)
        plt.imshow(image, cmap="gray")
        ax = fig.add_subplot(1, 2, 2)
        n, bins, patches = plt.hist(x=image.ravel(), rwidth=0.85, bins=128, range=(0, 1), log=True)
        plt.xlabel('Pixel intensity')
        plt.ylabel('Count')
        plt.title('Log histogram of pixel intensities ' + dataset.files[ind].filename)
        maxfreq = n.max()
        # Set a clean upper y-axis limit.
        plt.ylim(top=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
        try:
            plt.savefig(dir + str(dataset.labels[ind]) + "---" + dataset.files[ind].filename)
        except ValueError:
            print("Failed to write to file " + dir + str(dataset.labels[ind]) + "---" + dataset.files[ind].filename)


def compare_two_images(img1,img2, dir="../data_preprocessing/visualized/comparison/", filename="image.jpg"):
    # write two images side by side to a file for visual comparison
    assert(img1.shape == img2.shape)
    border = np.zeros([img1.shape[0], 3])
    combined_img = np.append(img1, border, axis=1)
    combined_img = np.append(combined_img, img2, axis=1)
    path = dir+filename
    #try:
    imsave(path, combined_img)
    #except ValueError:
        #print("Failed to write to file " + path)

def compare_preprocessing(preprocessing1, preprocessing2, dataset, n=50):
    for ind, _ in enumerate(dataset.images):
        image = imread(dataset.files[ind].path)
        img1 = preprocessing1(image)
        img2 = preprocessing2(image)
        compare_two_images(img1, img2, filename=str(ind)+".jpg")
        if n > 0 and ind > n:
            break

# histograms of labels in datasets
# for set in datasets:
#     visualize_labels(set)
# visualize_labels(DATA)


# visualize preprocessing
visualize_preprocessing(DATA,5)

#visualize_mean_image(DATA)
#show_average_pixel_histogram(DATA)

#for i in range(100):
#    i_rand = np.random.randint(0,len(DATA.images)-1)
#    show_pixel_histogram(DATA.images[i_rand], "image "+str(i_rand))

# visualize all pixel intensities
#save_pixel_intensities_to_files(DATA)

#compare_preprocessing(preprocess.preprocess, preprocess.preprocess_old, DATA)


# other metrics
print("mean of train data_preprocessing")
print(np.mean(DATA.labels))
print("mse if mean taken (for train)")
print(np.sum((DATA.labels - np.mean(DATA.labels)) ** 2) / np.shape(DATA.labels)[0])

print("variance of train data_preprocessing")
print(np.var(DATA.labels))


