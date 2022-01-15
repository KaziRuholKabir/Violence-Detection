import os
import cv2
from tqdm import tqdm


output_path = './Dataframes/hockey/'
IMG_SIZE = 224

def extract_frames(folder):
    
    for folders in os.listdir(folder):
        c = 0.00001
        print(folders)
        folder_path = os.path.join(folder, folders)
        for files in tqdm(os.listdir(folder_path)):
            path = os.path.join(folder_path, files)
            cap = cv2.VideoCapture(path)
            success = True

            while success:
                success, image = cap.read()
                if not success:
                    break

                cv2.resize(cv2.imwrite(output_path+ folders + str.format('{0:.5f}', c)[2:]+ '.jpg', image), (IMG_SIZE, IMG_SIZE))
                c += 0.00001
        print(c)
        print('Done: ' + folders)    

extract_frames('./Hockey_fights/Videos/')
