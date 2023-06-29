from dataset.JointsDataset import JointsDataset
import os
from natsort import natsorted
from termcolor import colored
import cv2
from PIL import Image
import torch
import numpy as np

VIDEO_LENGH_LIMIT = {
    # adroit
    "hammer-v0": 150,
    "pen-v0": 100,
    # "relocate-v0": 100,
    "door-v0": 150,

    # dexmv
    "pour": 200,
    "place_inside": 200,
    "relocate-mug": 100,
    "relocate-foam_brick": 100,
    "relocate-large_clamp":100, 
    "relocate-mustard_bottle":100,
    "relocate-potted_meat_can":100, 
    "relocate-sugar_box":100,
    "relocate-tomato_soup_can":100,

}

class RoboticDataset(JointsDataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None, task_name="none"):
        super().__init__(cfg, root, image_set, is_train, transform)

        self.transform = transform
        
        if task_name == "none": # multi-task 
            self.task_list = cfg.DATASET.TASK_LIST
        else: # single-task
            self.task_list = [task_name]

        print(colored('[RoboticDataset] Task list: {}'.format(self.task_list), 'cyan'))
        self.video_dirs = []
        self.video_length_limit = 0
        for task in self.task_list:
            task_dir = os.path.join(self.root, task)
            video_dirs = [os.path.join(task_dir, d) for d in os.listdir(task_dir) if os.path.isdir(os.path.join(task_dir, d))]
            video_dirs = natsorted(video_dirs)
            self.video_dirs += video_dirs
            self.video_length_limit = max(self.video_length_limit, VIDEO_LENGH_LIMIT[task])
        print(colored('[RoboticDataset] Video length limit: {}'.format(self.video_length_limit), 'cyan'))
        self.video_dirs = natsorted(self.video_dirs)
        
        # collect all video dirs, saving training time
        self.video_img_dirs = []
        for video_dir in self.video_dirs:
            img_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if os.path.isfile(os.path.join(video_dir, f))]
            img_files = natsorted(img_files)
            img_files = img_files[:self.video_length_limit]
            self.video_img_dirs.append(img_files)

        self.length = len(self.video_dirs)
        self.idx_range = len(self.video_dirs)
        print(colored('[RoboticDataset] Total {} videos'.format(len(self.video_img_dirs)), 'cyan'))


    def __getitem__(self, index):
        
        # video_dir = self.video_dirs[index]
        # # randomly get 2 imgs
        # img_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if os.path.isfile(os.path.join(video_dir, f))]
        # img_files = natsorted(img_files)

        img_files = self.video_img_dirs[index]
        src_idx, tgt_idx = np.random.choice(len(img_files), 2, replace=False)

        img_path_src = img_files[src_idx]
        img_path_tgt = img_files[tgt_idx]

        
        def read_img(img_path):
            data_numpy = cv2.imread(
                img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

            if self.color_rgb:
                data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
    
            if data_numpy is None:
                raise ValueError('Fail to read {}'.format(img_path))

            img = Image.fromarray(data_numpy)

            if self.transform is not None:
                img = self.transform(img)
            
            return img
        
        img1 = read_img(img_path_src)
        img2 = read_img(img_path_tgt)
            

        # stack two imgs
        img = torch.stack((img1, img2), dim=0)

        target, target_weight, meta = torch.zeros(1), torch.zeros(1), {}

        return img, target, target_weight, meta

    def __len__(self):
        return len(self.video_dirs)
        