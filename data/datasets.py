# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
from torchvision import transforms
from torch.utils.data import Dataset

from PIL import Image
import io
import torch
from .dct import DCT_base_Rec_Module
import random

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import kornia.augmentation as K

Perturbations = K.container.ImageSequential(
    K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 3.0), p=0.1),
    K.RandomJPEG(jpeg_quality=(30, 100), p=0.1)
)

transform_before = transforms.Compose([
    # transforms.Resize([256, 256]),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: Perturbations(x)[0])
    ]
)
transform_before_test = transforms.Compose([
    transforms.ToTensor(),
    ]
)

transform_train = transforms.Compose([
    transforms.Resize([256, 256]),
    # transforms.Lambda(lambda x: Perturbations(x)[0]),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)

transform_test_normalize = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)

class RandomJPEG():
    def __init__(self, quality=95, interval=1, p=0.1):
        if isinstance(quality, tuple):
            self.quality = [i for i in range(quality[0], quality[1]) if i % interval == 0]
        else:
            self.quality = quality
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            if isinstance(self.quality, list):
                quality = random.choice(self.quality)
            else:
                quality = self.quality
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            img = Image.open(buffer)
        return img


class RandomGaussianBlur():
    def __init__(self, kernel_size, sigma=(0.1, 2.0), p=1.0):
        self.blur = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return self.blur(img)
        return img


class RandomMask(object):
    def __init__(self, ratio=0.5, patch_size=16, p=0.5):
        """
        Args:
            ratio (float or tuple of float): If float, the ratio of the image to be masked.
                                             If tuple of float, random sample ratio between the two values.
            patch_size (int): the size of the mask (d*d).
        """
        if isinstance(ratio, float):
            self.fixed_ratio = True
            self.ratio = (ratio, ratio)
        elif isinstance(ratio, tuple) and len(ratio) == 2 and all(isinstance(r, float) for r in ratio):
            self.fixed_ratio = False
            self.ratio = ratio
        else:
            raise ValueError("Ratio must be a float or a tuple of two floats.")

        self.patch_size = patch_size
        self.p = p

    def __call__(self, tensor):

        if random.random() > self.p: return tensor

        _, h, w = tensor.shape
        mask = torch.ones((h, w), dtype=torch.float32)

        if self.fixed_ratio:
            ratio = self.ratio[0]
        else:
            ratio = random.uniform(self.ratio[0], self.ratio[1])

        # Calculate the number of masks needed
        num_masks = int((h * w * ratio) / (self.patch_size ** 2))

        # Generate non-overlapping random positions
        selected_positions = set()
        while len(selected_positions) < num_masks:
            top = random.randint(0, (h // self.patch_size) - 1) * self.patch_size
            left = random.randint(0, (w // self.patch_size) - 1) * self.patch_size
            selected_positions.add((top, left))

        for (top, left) in selected_positions:
            mask[top:top+self.patch_size, left:left+self.patch_size] = 0

        return tensor * mask.expand_as(tensor)


def Get_Transforms(args):

    size = args.input_size

    TRANSFORM_DICT = {
        'resize_BILINEAR': {
            'train': [
                transforms.RandomResizedCrop([size, size], interpolation=InterpolationMode.BILINEAR),
            ],
            'eval': [
                transforms.Resize([size, size], interpolation=InterpolationMode.BILINEAR),
            ],
        },

        'resize_NEAREST': {
            'train': [
                transforms.RandomResizedCrop([size, size], interpolation=InterpolationMode.NEAREST),
            ],
            'eval': [
                transforms.Resize([size, size], interpolation=InterpolationMode.NEAREST),
            ],
        },

        'crop': {
            'train': [
                transforms.RandomCrop([size, size], pad_if_needed=True),
            ],
            'eval': [
                transforms.CenterCrop([size, size]),
            ],
        },

        'source': {
            'train': [
                transforms.RandomCrop([size, size], pad_if_needed=True),
            ],
            'eval': [
            ],
        },
    }

    # region [Augmentations]
    safe_transform_train, safe_transform_eval = TRANSFORM_DICT[args.transform_mode]['train'], TRANSFORM_DICT[args.transform_mode]['eval']

    safe_transform_train.extend([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(180),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        transforms.ToTensor(),
        RandomMask(ratio=(0.00, 0.75), patch_size=16, p=0.5),
    ])

    safe_transform_eval.append(transforms.ToTensor())
    # endregion

    # region [Perturbatiocns in Testing]
    if args.jpeg_factor is not None:
        safe_transform_eval.insert(0, RandomJPEG(quality=args.jpeg_factor, p=1.0))
    if args.blur_sigma is not None:
        safe_transform_eval.insert(0, transforms.GaussianBlur(kernel_size=5, sigma=args.blur_sigma))
    if args.mask_ratio is not None and args.mask_patch_size is not None:
        safe_transform_eval.append(RandomMask(ratio=args.mask_ratio, patch_size=args.mask_patch_size, p=1.0))
    # endregion

    return transforms.Compose(safe_transform_train), transforms.Compose(safe_transform_eval)

class TrainDataset(Dataset):
    def __init__(self, is_train, args):
        
        TRANSFORM = Get_Transforms(args)
        self.transform = TRANSFORM[0] if is_train else TRANSFORM[1]
        root = args.data_path if is_train else args.eval_data_path

        self.data_list = []

        if'GenImage' in root and root.split('/')[-1] != 'train':
            file_path = root

            if '0_real' not in os.listdir(file_path):
                for folder_name in os.listdir(file_path):
                
                    # assert os.listdir(os.path.join(file_path, folder_name)) == ['0_real', '1_fake']
                    assert os.listdir(os.path.join(file_path, folder_name)) == ['1_fake', '0_real']

                    for image_path in os.listdir(os.path.join(file_path, folder_name, '0_real')):
                        self.data_list.append({"image_path": os.path.join(file_path, folder_name, '0_real', image_path), "label" : 0})
                 
                    for image_path in os.listdir(os.path.join(file_path, folder_name, '1_fake')):
                        self.data_list.append({"image_path": os.path.join(file_path, folder_name, '1_fake', image_path), "label" : 1})
            
            else:
                for image_path in os.listdir(os.path.join(file_path, '0_real')):
                    self.data_list.append({"image_path": os.path.join(file_path, '0_real', image_path), "label" : 0})
                for image_path in os.listdir(os.path.join(file_path, '1_fake')):
                    self.data_list.append({"image_path": os.path.join(file_path, '1_fake', image_path), "label" : 1})
        else:

            for filename in os.listdir(root):

                file_path = os.path.join(root, filename)

                if '0_real' not in os.listdir(file_path):
                    for folder_name in os.listdir(file_path):
                    
                        assert os.listdir(os.path.join(file_path, folder_name)) == ['1_fake', '0_real']
                        # assert os.listdir(os.path.join(file_path, folder_name)) == ['0_real', '1_fake']

                        for image_path in os.listdir(os.path.join(file_path, folder_name, '0_real')):
                            self.data_list.append({"image_path": os.path.join(file_path, folder_name, '0_real', image_path), "label" : 0})
                    
                        for image_path in os.listdir(os.path.join(file_path, folder_name, '1_fake')):
                            self.data_list.append({"image_path": os.path.join(file_path, folder_name, '1_fake', image_path), "label" : 1})
                
                else:
                    for image_path in os.listdir(os.path.join(file_path, '0_real')):
                        self.data_list.append({"image_path": os.path.join(file_path, '0_real', image_path), "label" : 0})
                    for image_path in os.listdir(os.path.join(file_path, '1_fake')):
                        self.data_list.append({"image_path": os.path.join(file_path, '1_fake', image_path), "label" : 1})
                
        self.dct = DCT_base_Rec_Module()


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        
        sample = self.data_list[index]
                
        image_path, targets = sample['image_path'], sample['label']

        try:
            image = Image.open(image_path).convert('RGB')
        except:
            print(f'image error: {image_path}')
            return self.__getitem__(random.randint(0, len(self.data_list) - 1))


        x_0 = self.transform(image)
        image = transform_before(image)

        try:
            x_minmin, x_maxmax, x_minmin1, x_maxmax1 = self.dct(image)
        except:
            print(f'image error: {image_path}, c, h, w: {image.shape}')
            return self.__getitem__(random.randint(0, len(self.data_list) - 1))

        # x_0 = transform_train(image)
        x_minmin = transform_train(x_minmin) 
        x_maxmax = transform_train(x_maxmax)

        x_minmin1 = transform_train(x_minmin1) 
        x_maxmax1 = transform_train(x_maxmax1)
        


        return torch.stack([x_minmin, x_maxmax, x_minmin1, x_maxmax1, x_0], dim=0), torch.tensor(int(targets))

    

class TestDataset(Dataset):
    def __init__(self, is_train, args):
        
        TRANSFORM = Get_Transforms(args)
        self.transform = TRANSFORM[0] if is_train else TRANSFORM[1]
        root = args.data_path if is_train else args.eval_data_path

        self.data_list = []

        file_path = root

        if '0_real' not in os.listdir(file_path):
            for folder_name in os.listdir(file_path):
    
                # assert os.listdir(os.path.join(file_path, folder_name)) == ['0_real', '1_fake']
                assert os.listdir(os.path.join(file_path, folder_name)) == ['1_fake', '0_real']
                
                for image_path in os.listdir(os.path.join(file_path, folder_name, '0_real')):
                    self.data_list.append({"image_path": os.path.join(file_path, folder_name, '0_real', image_path), "label" : 0})
                
                for image_path in os.listdir(os.path.join(file_path, folder_name, '1_fake')):
                    self.data_list.append({"image_path": os.path.join(file_path, folder_name, '1_fake', image_path), "label" : 1})
        
        else:
            for image_path in os.listdir(os.path.join(file_path, '0_real')):
                self.data_list.append({"image_path": os.path.join(file_path, '0_real', image_path), "label" : 0})
            for image_path in os.listdir(os.path.join(file_path, '1_fake')):
                self.data_list.append({"image_path": os.path.join(file_path, '1_fake', image_path), "label" : 1})


        self.dct = DCT_base_Rec_Module()


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        
        sample = self.data_list[index]
                
        image_path, targets = sample['image_path'], sample['label']

        image = Image.open(image_path).convert('RGB')
        x_0 = self.transform(image)

        image = transform_before_test(image)

        # x_max, x_min, x_max_min, x_minmin = self.dct(image)

        x_minmin, x_maxmax, x_minmin1, x_maxmax1 = self.dct(image)


        # x_0 = transform_train(image)
        x_minmin = transform_train(x_minmin) 
        x_maxmax = transform_train(x_maxmax)

        x_minmin1 = transform_train(x_minmin1) 
        x_maxmax1 = transform_train(x_maxmax1)
        
        return torch.stack([x_minmin, x_maxmax, x_minmin1, x_maxmax1, x_0], dim=0), torch.tensor(int(targets))

