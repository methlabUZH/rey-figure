import numpy as np
from config import MODEL_PATH, DATA_DIR, LABEL_FORMAT
from model import CNN
from cv2 import imread, imwrite
import cv2
from preprocess import preprocess, preprocess_basic
import os
import csv
import matplotlib.pyplot as plt

# adapt such that it works for all formats ('one-per-item', ...)
# restore commit in result.txt to ensure loading model parameters works
def restore(pathname):
    model = CNN('restored', 0)
    model.restore_model(pathname)
    return model

score_path=DATA_DIR + 'scored/'
if not os.path.exists(score_path):
    os.mkdir(score_path)
if __name__ == "__main__":
    model = restore(MODEL_PATH)

    # images in folder ../new_data/testdata
    path = DATA_DIR + 'testdata/'
    score_file = DATA_DIR + "scored" + '.csv'

    filenames = []
    scores = []
    rscores = []
    if os.path.exists(score_file):
        print("Reading allready scored images")
        with open(score_file) as csv_file:
            labels_reader = csv.reader(csv_file, delimiter=',')
            rows = [row for row in labels_reader]
            filenames = [row[0] for row in rows]
            scores = [float(row[1]) for row in rows]
            rscores = [int(row[2]) for row in rows]



    while True:
        filename =input("Which do you want to score next? ")
        if filename=="Exit":
            print("Exiting the process")
            break
        if not os.path.exists(path+filename+'.jpg'):
            print("Invalid image provided")
            continue
        img=imread(path+filename+'.jpg')
        img2 = preprocess_basic(img)
        img = preprocess(img)
        img=np.expand_dims(img, 0)
        results = model.predict(img)

        height, width = img.shape[1:]

        new_width, new_height = int(width + np.ceil(width/2)), int(height + np.ceil(height))


        # Create a new canvas with new width and height.
        canvas = np.ones((new_height, new_width), dtype=np.uint8) * 255
        # New replace the center of canvas with original image
        padding_top, padding_left = 60, 10
        if padding_top + height < new_height and padding_left + width < new_width:
            # preprocessing changes black and white encoding
            canvas[padding_top:padding_top + height, padding_left:padding_left + width] = 255*img2
        else:
            print("The Given padding exceeds the limits.")
        if LABEL_FORMAT == 'one-per-item':
            score = results[0, 18]

        else:
            score = results[0, 0]
        rscore = np.clip(int(np.round(score)), 0, 36)
        text1 = "Score: {}".format(rscore)
        print("Provided image scored {}".format(rscore))
        rscores.append(rscore)
        scores.append(score)


        final = cv2.putText(canvas.copy(), text1, (int(0.25 * width), 30), cv2.FONT_HERSHEY_COMPLEX, 1, 0)

        plt.figure()
        plt.imshow(final, cmap='Greys_r')
        plt.show(block=False)
        plt.pause(4)
        plt.close()
        imwrite(score_path + 'scored_' + filename +'.jpg', final)
        filenames.append(filename)
        ind = np.argsort(-np.array(scores))
        filenames2 = np.array(filenames)[ind]
        rscores2 = np.array(rscores)[ind]
        print ("Current Leaderboard")
        for i in range(min(10,np.shape(filenames2)[0])):
            print("{}: {} with score of {}".format(i+1,filenames2[i],rscores2[i]))
        print("\n \n \n")

        with open(score_file, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(zip(filenames, scores,rscores))
        csv_file.close()



