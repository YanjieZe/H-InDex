from dataset.JointsDataset import JointsDataset
import os
from natsort import natsorted
from termcolor import colored
import cv2
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm



class RoboTubeDataset(JointsDataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        super().__init__(cfg, root, image_set, is_train, transform)

        self.transform = transform
        
        # multi-task 
        self.task_list = cfg.DATASET.TASK_LIST
        print(colored('[RoboticDataset] Task list: {}'.format(self.task_list), 'cyan'))
        self.video_dirs = []
        self.video_interval = 5

        for task in self.task_list:
            for sub1 in os.listdir(os.path.join(self.root, task)):
                for sub2 in os.listdir(os.path.join(self.root, task, sub1)):
                    for sub3 in os.listdir(os.path.join(self.root, task, sub1, sub2)):
                        for sub4 in os.listdir(os.path.join(self.root, task, sub1, sub2, sub3)):
                            for sub5 in os.listdir(os.path.join(self.root, task, sub1, sub2, sub3, sub4)):
                                self.video_dirs.append(os.path.join(self.root, task, sub1, sub2, sub3, sub4, sub5))

            
        self.video_dirs = natsorted(self.video_dirs)
        
        # collect all video dirs, saving training time
        self.video_img_dirs = []
        for video_dir in tqdm(self.video_dirs, desc="Collecting video dirs"):
            img_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if os.path.isfile(os.path.join(video_dir, f)) and f.endswith("_color.png")]
            img_files = natsorted(img_files)
            if len(img_files) ==0:
                print(f"video_dir {video_dir} is empty. pass")
                # del video_dir
                self.video_dirs.remove(video_dir)
                continue
            img_files = img_files[::self.video_interval]
            self.video_img_dirs.append(img_files)

        self.length = len(self.video_img_dirs)
        self.idx_range = len(self.video_img_dirs)
        print(colored('[RoboticDataset] Total {} videos'.format(len(self.video_img_dirs)), 'cyan'))
        num_img = sum (len(img_files) for img_files in self.video_img_dirs)
        print(colored('[RoboticDataset] Total {} images'.format(num_img), 'cyan'))


    def __getitem__(self, index):
        
        # video_dir = self.video_dirs[index]
        # # randomly get 2 imgs
        # img_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if os.path.isfile(os.path.join(video_dir, f))]
        # img_files = natsorted(img_files)

        img_files = self.video_img_dirs[index]
        try:
            src_idx, tgt_idx = np.random.choice(len(img_files), 2, replace=False)
        except:
            print("len(img_files): ", len(img_files))
            import ipdb; ipdb.set_trace()

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
        return len(self.video_img_dirs)
        