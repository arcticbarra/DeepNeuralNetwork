import os

base = "hd_dataset/test/hot_dog"

for filename in os.listdir(base):
    dst = "/hotdog" + filename
    src = base + "/" + filename
    dst = base + dst

    # rename() function will
    # rename all the files
    os.rename(src, dst)
