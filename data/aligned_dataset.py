import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGBA')
        # split AB image into A and B
        w, h = AB.size
        A0 = np.array(AB.crop((0, 0, 224, h)))
        A1 = np.array(AB.crop((224, 0, 448, h)))
        B = AB.crop((448, 0, w, h)).convert('RGB') #just 3 channels for B
        B = np.array(B)
        print(f'B shape: {B.shape}')
        print(B[0][0][0]) #to check if same B is being used each time (it shouldn't be)
        B = Image.fromarray(B.astype('uint8'), 'RGB')

        print(f'A0 shape: {A0.shape}')
        print(f'A1 shape: {A1.shape}')

        #split A into 8 separate greyscale images
        greyscale_ims = []
        for arr in [A0, A1]:
            for i in range(arr.shape[2]):
                greyscale_im = Image.fromarray(arr[:, :, i].astype('uint8'))
                greyscale_ims.append(greyscale_im)
            

        '''
        A = np.hstack((A0,A1)) #np.stack(.. axis=2)
        A = Image.fromarray(A.astype('uint8'), 'RGB')
        '''


        # apply the same transform to both A and B
        transform_params = get_params(self.opt, greyscale_ims[0].size)
        #A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        A_transform = get_transform(self.opt, transform_params, grayscale=(True))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        for i, greyscale_im in enumerate(greyscale_ims):
            greyscale_ims[i] = A_transform(greyscale_im)

        B = B_transform(B)

        #convert A images back into a single numpy array
        for i, greyscale_im in enumerate(greyscale_ims):
            if i == 0:
                A = greyscale_im
            else:
                A = np.concatenate((A, greyscale_im), axis=0)

        #turn B into a numpy array too (to match A)
        B = np.array(B)
        
        print('')
        print(f'A SHAPE after concatenation: {A.shape}')
        print(f'final B SHAPE: {B.shape}')
        print

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
