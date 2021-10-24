import numpy as np
from config import MODEL_PATH, DATA_DIR, LABEL_FORMAT, DEST_PATH, DATA_AUGMENTATION
from model import CNN
from cv2 import imread, imwrite
import cv2
from preprocess import preprocess, preprocess_basic
from augmenter import simulate_augment

# adapt such that it works for all formats ('one-per-item', ...)
# restore commit in result.txt to ensure loading model parameters works


def restore(pathname):
    model = CNN('restored', 0)
    model.restore_model(pathname)
    return model


def getTestImages(foldername):
    path = DATA_DIR + foldername + '/'
    filenames = np.genfromtxt('../data_preprocessing/' + foldername + '.csv', dtype='|U64')
    return path, filenames


def predict_restored(models, img_path, name):

    img = imread(img_path)
    raw_image = preprocess_basic(img)
    image = preprocess(img)
    if DATA_AUGMENTATION:
        image = simulate_augment(image)

    image = np.asarray([image])

    results = []

    for i in range(10):
        result = models[i].predict(image)
        if LABEL_FORMAT == 'one-per-item':
            results.append(result[0, 18])
        elif LABEL_FORMAT == 'one':
            results.append(result)

    result = np.mean(np.asarray(results))

    height, width = raw_image.shape

    new_width, new_height = int(width + np.ceil(width / 4)), int(height + np.ceil(height/4))

    # Create a new canvas with new width and height.
    canvas = np.ones((new_height, new_width), dtype=np.uint8) * 255

    # New replace the center of canvas with original image
    padding_top, padding_left = 60, 10
    if padding_top + height < new_height and padding_left + width < new_width:
        canvas[padding_top:padding_top + height, padding_left:padding_left + width] = 255 * raw_image
    else:
        print("The Given padding exceeds the limits.")

    score = (np.clip(round(2*result)/2, 0, 36))
    text1 = "Score: {}".format(score)

    final = cv2.putText(canvas.copy(), text1, (int(0.25 * width), 120), cv2.FONT_HERSHEY_COMPLEX, 5, 0, cv2.LINE_8)

    if DEST_PATH is None:
        print("DEST_PATH is required to run demo. Please update config file...")
        exit(1)
    imwrite(DEST_PATH + ".".join(name.split(".")[0:-1]) + "---" + str(score) + "---" + str(result) + "." + name.split(".")[-1], final)


if __name__ == "__main__":
    # model path contains folder where models are stored
    model = restore(MODEL_PATH)

    # images in folder ../data_preprocessing/testdata
    path, filenames = getTestImages('testdata')
    images = []
    raw_images = []

    for image in range(filenames.shape[0]):
        print(path + filenames[image])
        img = imread(path + filenames[image])
        raw_images.append(preprocess_basic(img))
        img = preprocess(img)
        images.append(img)

    results = model.predict(images)

    for i in range(results.shape[0]):
        height, width = images[i].shape

        new_width, new_height = int(width + np.ceil(width/2)), int(height + np.ceil(height))


        # Create a new canvas with new width and height.
        canvas = np.ones((new_height, new_width), dtype=np.uint8) * 255

        # New replace the center of canvas with original image
        padding_top, padding_left = 60, 10
        if padding_top + height < new_height and padding_left + width < new_width:
            # preprocessing changes black and white encoding
            canvas[padding_top:padding_top + height, padding_left:padding_left + width] = 255 * raw_images[i]
        else:
            print("The Given padding exceeds the limits.")
        if LABEL_FORMAT == 'one-per-item':
            text1 = "Score: {}".format(int(results[i, 18]))
        else:
            text1 = "Score: {}".format(int(results[i]))

        final = cv2.putText(canvas.copy(), text1, (int(0.25 * width), 30), cv2.FONT_HERSHEY_COMPLEX, 1, 0)

        imwrite(DEST_PATH + 'scored_' + filenames[i], final)
