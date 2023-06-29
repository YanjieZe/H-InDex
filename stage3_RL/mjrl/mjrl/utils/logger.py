import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy
import pickle
import os
import csv
from termcolor import colored, cprint

class DataLog:

    def __init__(self, use_wandb=False):
        self.log = {}
        self.max_len = 0
        self.use_wandb = use_wandb
        if self.use_wandb:
            import wandb
            self.wandb = wandb
            print(colored(f"[wandb] wandb is used in logger.", "cyan"))

    def log_kv(self, key, value, step=None):
        # logs the (key, value) pair

        # TODO: This implementation is error-prone:
        # it would be NOT aligned if some keys are missing during one iteration.
        if key not in self.log:
            self.log[key] = []
        self.log[key].append(value)
        if len(self.log[key]) > self.max_len:
            self.max_len = self.max_len + 1
        
    def log_dict(self, dict, step=None):
        if self.use_wandb:
            self.wandb.log(dict, step=step)

    def log_video(self, key, video, step=None):
        if self.use_wandb:
            self.wandb.log({key: self.wandb.Video(video, fps=25, format="mp4")}, step=step)
    
    def log_ckpt(self, ckpt, step=None):
        if self.use_wandb:
            self.wandb.save(ckpt)
            cprint(f"[wandb] ckpt is logged.", "cyan")
            
    def save_log(self, save_path):
        # TODO: Validate all lengths are the same.
        pickle.dump(self.log, open(save_path + '/log.pickle', 'wb'))
        with open(save_path + '/log.csv', 'w') as csv_file:
            fieldnames = list(self.log.keys())
            if 'iteration' not in fieldnames:
                fieldnames = ['iteration'] + fieldnames

            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in range(self.max_len):
                row_dict = {'iteration': row}
                for key in self.log.keys():
                    if row < len(self.log[key]):
                        row_dict[key] = self.log[key][row]
                writer.writerow(row_dict)

    def get_current_log(self):
        row_dict = {}
        for key in self.log.keys():
            # TODO: this is very error-prone (alignment is not guaranteed)
            row_dict[key] = self.log[key][-1]
        return row_dict

    def shrink_to(self, num_entries):
        for key in self.log.keys():
            self.log[key] = self.log[key][:num_entries]

        self.max_len = num_entries
        assert min([len(series) for series in self.log.values()]) == \
            max([len(series) for series in self.log.values()])

    def read_log(self, log_path):
        assert log_path.endswith('log.csv')

        with open(log_path) as csv_file:
            reader = csv.DictReader(csv_file)
            listr = list(reader)
            keys = reader.fieldnames
            data = {}
            for key in keys:
                data[key] = []
            for row, row_dict in enumerate(listr):
                for key in keys:
                    try:
                        data[key].append(eval(row_dict[key]))
                    except:
                        print("ERROR on reading key {}: {}".format(key, row_dict[key]))

                if 'iteration' in data and data['iteration'][-1] != row:
                    raise RuntimeError("Iteration %d mismatch -- possibly corrupted logfile?" % row)

        self.log = data
        self.max_len = max(len(v) for k, v in self.log.items())
        print("Log read from {}: had {} entries".format(log_path, self.max_len))
