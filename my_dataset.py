import os
from PIL import Image
from torch.utils.data import Dataset


class myDataset(Dataset):
    def __init__(self, data_dir, transform):
        # data_dir：图片保存路径
        # transform：初始化方法
        self.data_dir = data_dir
        self.transform = transform
        self.img_names = [name for name in list(filter(lambda x: x.endswith(".png"), os.listdir(self.data_dir)))]

    def __getitem__(self, index):
        path_img = os.path.join(self.data_dir, self.img_names[index])
        img = Image.open(path_img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        if len(self.img_names) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images_wgan_gp!".format(self.data_dir))
        return len(self.img_names)
