import os
import time
from multiprocessing import Pool


ORIG_IMAGES_DIR = "images/semantic_drone_dataset/original_images"
ORIG_LABELS_DIR = "images/semantic_drone_dataset/label_images_semantic"

pairs = []
for idx, img in enumerate(os.listdir(ORIG_IMAGES_DIR)):
    lab = img.strip(".jpg") + ".png"
    print(idx, (img, lab))



# def do_shit(tup):
#     print(tup[0])
#     time.sleep(1)
#     print(tup[1])


# if __name__ == "__main__":
#     list_of_everything = [
#         ("start 1", "end 1"),
#         ("start 2", "end 2"),
#         ("start 3", "end 3"),
#         ("start 4", "end 4"),
#         ("start 5", "end 5"),
#         ("start 6", "end 6"),
#         ("start 7", "end 7"),
#         ("start 8", "end 8"),
#         ("start 9", "end 9"),
#         ("start 10", "end 10"),
#         ("start 11", "end 11"),
#         ("start 12", "end 12"),
#         ("start 13", "end 13"),
#         ("start 14", "end 14"),
#         ("start 15", "end 15")
#     ]

#     with Pool() as pool:
#         pool.map(do_shit, list_of_everything)
