import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence


class Encoder(nn.Module):

    def __init__(self, embedding_dim=64, hidden_dim=64, mlp_dim=1024, num_layer=1, dropout=0.0, obs_lack='disable'):
        super(Encoder, self).__init__()

        self.mlp_dim = mlp_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_layer = num_layer
        self.obs_lack = obs_lack

        self.encoder = nn.LSTM(embedding_dim, hidden_dim, num_layer, dropout=dropout)
        self.embedding = nn.Linear(2, embedding_dim)

    def forward(self, batch_data):
        # 行人轨迹序列输入长度不一，则我们pack_sequence不同长度的序列
        if self.obs_lack == 'enable':
            pass
        # 历史轨迹长obs_len, 预测轨迹长pred_len
        elif self.obs_lack == 'disable':
            batch = batch_data.size(1)
            obs_traj_embedding = self.embedding(batch_data.view(-1, 2))
            obs_traj_embedding = obs_traj_embedding.view(-1, batch, self.embedding_dim)
            init_hidden = torch.zeros(self.num_layers, batch, self.h_dim).cuda()
            output, state = self.encoder(obs_traj_embedding, (init_hidden, init_hidden))
            return state[0]


class Decoder(nn.Module):

    def __init__(self, seq_len, embedding_dim=64, h_dim=128, mlp_dim=1024, num_layers=1,
                 dropout=0.0, activation='Relu', batch_norm=True):
        super(Decoder, self).__init__()
        self.seq_len = seq_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim

        self.decoder = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)

        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.hidden2pos = nn.Linear(h_dim, 2)

    def forward(self, batch_data):
        pass


class VanillaLSTM(nn.Module):

    def __init__(self, args):
        super(VanillaLSTM, self).__init__()
        self.encoder = Encoder(args['Embedding']['Input_size'],
                               args['LSTMs']['Hidden_size'],
                               args['MLP']['MLP_dim'],
                               args['LSTMs']['Num_layer'],
                               args['Embedding'])

    def forward(self, batch_data):
        pass
