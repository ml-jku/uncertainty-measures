import os
import wget
import shutil

from torchvision import transforms

from source.constants import CIFAR_MEAN, CIFAR_STD

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD,)
])

train_transform = transforms.Compose([
    transforms.RandomCrop((32, 32), padding=4),
    transforms.RandomHorizontalFlip(),
    transform
])

# stack resize on top of other transforms for Tiny ImageNet
tin_transform = lambda t: transforms.Compose([transforms.Resize((32, 32)), t])

def download_tiny_imagenet(data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    if not os.path.exists(os.path.join(data_dir, "tiny-imagenet-200.zip")):
        wget.download("http://cs231n.stanford.edu/tiny-imagenet-200.zip", os.path.join(data_dir, "tiny-imagenet-200.zip"))
    if not os.path.exists(os.path.join(data_dir, "tiny-imagenet-200")):
        print("unpacking...")
        shutil.unpack_archive(os.path.join(data_dir, "tiny-imagenet-200.zip"), os.path.join(data_dir))
        print("unpacked")

    # move images in val folder to separate subfolders
    if os.path.exists(os.path.join(data_dir, "tiny-imagenet-200", "val", "images")):
        print("moving validation set according to annotations...")
        path = os.path.join(data_dir, "tiny-imagenet-200", "val")
        
        with open(os.path.join(path, "val_annotations.txt")) as f:
            files = dict()
            for line in f:
                split = line.split("\t")
                files[split[0]] = split[1]

        for file, folder in files.items():
            folder_path = os.path.join(path, folder)
            os.makedirs(folder_path, exist_ok=True)
            shutil.move(os.path.join(path, "images", file), os.path.join(folder_path, file))
        
        os.rmdir(os.path.join(path, "images"))
        print("moved")
    

def download_lsun(data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    if not os.path.exists(os.path.join(data_dir, "LSUN_resize.tar.gz")):
        wget.download("https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz?dl=1", os.path.join(data_dir, "LSUN_resize.tar.gz"))
    if not os.path.exists(os.path.join(data_dir, "LSUN_resize")):
        print("unpacking...")
        shutil.unpack_archive(os.path.join(data_dir, "LSUN_resize.tar.gz"), os.path.join(data_dir))
        print("unpacked")
