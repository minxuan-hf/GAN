# 把所有图片转换成相同尺寸
from PIL import Image
import os.path
import glob
import os

os.makedirs("ROP_train445_resize224")


def Resize(file, outdir, width, height):
    imgFile = Image.open(file)
    try:
        newImage = imgFile.resize((width, height), Image.BILINEAR)
        newImage.save(os.path.join(outdir, os.path.basename(file)))
    except Exception as e:
        print(e)


for file in glob.glob("ROP_train445/*.png"):  # 图片所在的目录
    # 将图片转换成224x224的尺寸
    Resize(file, "ROP_train445_resize224", 224, 224)  # 新图片存放的目录,需要统一的尺寸
