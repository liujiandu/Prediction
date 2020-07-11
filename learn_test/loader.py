import os
import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence


def data_loader(args):
    data_set = TrajectoryDataset(
        args['Train'],
        obs_len=args['Obs_len'],
        pred_len=args['Pred_len'],
        delim=args['delim'],
        form=args['Data_form'],
        obs_lack=args['Obs_lack'])

    loader = DataLoader(
        data_set,
        batch_size=args['Batch_size'],
        shuffle=args['shuffle'],
        collate_fn=seq_collate,
        drop_last=False)

    return loader


def seq_collate(data):
    (obs_seq_list, pred_seq_list) = zip(*data)

    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    out = [obs_traj, pred_traj]

    return tuple(out)


def load_file(file_dir, delim='\t'):
    data = []
    with open(file_dir, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


def traj_to_graph():
    pass


class TrajectoryDataset(Dataset):
    def __init__(self, data_dir, obs_len=8, pred_len=12, feature_num=2, delim='\t', form='pedestrian', obs_lack='disable'):
        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.delim = delim
        self.num = 100
        self.seq_len = obs_len + pred_len

        all_files = [os.path.join(self.data_dir, _path) for _path in os.listdir(self.data_dir)]
        seq_list = []

        if form == 'pedestrian':
            for path in all_files:
                data = load_file(path, delim)
                frames = np.unique(data[:, 0])
                frame_data = []
                # 同frame_id放一起
                for frame in frames:
                    frame_data.append(data[frame == data[:, 0], :])
                # 每次抽取frame为seplen长的轨迹数据,按行人排列后抽取数据更方便，不过这种取法相比于按行人抽取更方便添加social pooling等计算
                for frame_idx in range(0, len(frame_data)-1, 1):
                    curr_seq_traj = np.concatenate(frame_data[frame_idx:frame_idx+self.seq_len], axis=0)
                    peds_in_curr = np.unique(curr_seq_traj[:, 1])
                    num_peds_loaded = 0
                    curr_seq = np.zeros((len(peds_in_curr), feature_num, self.seq_len))
                    for _, ped_id in enumerate(peds_in_curr):
                        curr_ped_traj = curr_seq_traj[curr_seq_traj[:, 1] == ped_id, :]
                        curr_ped_traj = np.around(curr_ped_traj, decimals=4)
                        ped_frame_start = frames.index(curr_ped_traj[0, 0]) - frame_idx
                        ped_frame_end = frames.index(curr_ped_traj[-1, 0]) - frame_idx + 1

                        if ped_frame_end - ped_frame_start != self.seq_len:
                            # 不允许历史轨迹不足则跳过该行人
                            if obs_lack == "disable":
                                continue
                            # 允许历史轨迹不足
                            elif obs_lack == "enable":
                                # 长度不足但轨迹未中断则pack
                                if curr_ped_traj[-1, 0] == frames[frame_idx+self.seq_len]:
                                    # 前部填充直到满足长度
                                    pass
                                # 剔除osb中轨迹中断的行人
                                else:
                                    continue
                        curr_ped_traj = np.transpose(curr_ped_traj[:, 2:])
                        ped_idx = num_peds_loaded
                        curr_seq[ped_idx, :, ped_frame_start:ped_frame_end] = curr_ped_traj
                        num_peds_loaded += 1
                    seq_list.append(curr_seq[:num_peds_loaded])
            self.num_seq = len(seq_list)
            seq_list = np.concatenate(seq_list, axis=0)
            self.obs_traj = torch.from_numpy(seq_list[:, :, :self.obs_len]).type(torch.float)
            self.pred_traj = torch.from_numpy(seq_list[:, :, self.obs_len:]).type(torch.float)

        elif form == 'frame':
            self.obs_traj = []
            self.pred_traj = []
            for path in all_files:
                data = load_file(path, delim)
                frames = np.unique(data[:, 0])
                frame_data = []
                # 同frame_id放一起
                for frame in frames:
                    frame_data.append(data[frame == data[:, 0], :])
                for frame_idx in range(0, len(frame_data)-self.seq_len-1):
                    curr_frame_traj = frame_data[frame_idx:frame_idx+self.seq_len]
                    self.obs_traj.append(curr_frame_traj[:self.obs_len])
                    self.pred_traj.append(curr_frame_traj[self.obs_len:])
            self.num_seq = len(self.obs_traj)

    def __len__(self):
        return self.num_seq

    def __getitem__(self, item):
        out = [torch.from_numpy(self.obs_traj[item]).type(torch.float),
               torch.from_numpy(self.pred_traj[item]).type(torch.float)]
        return out
