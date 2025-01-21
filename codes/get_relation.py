import torch
import torch.nn.functional as F
from torch import nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        #self.fc3 = nn.Linear()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x



class Graph_Attention(nn.Module):##Learning implicit relation

    def __init__(self, in_features, out_features, dropout, concat=True, residual=False, img_dim=1024, h_dim=500,tail_rel=None):
        super(Graph_Attention, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        #self.alpha = alpha

        self.concat = concat
        self.residual = residual
        #tail_rel = tail_rel.shape[0]

        #使用卷积提取
        # self.seq_transformation_r_h = nn.Conv1d(in_features, 1, kernel_size=1, stride=1, bias=False)
        # self.seq_transformation_r_t = nn.Conv1d(in_features, 1, kernel_size=1, stride=1, bias=False)

        #使用mlp提取
        self.mlp_r_h = MLP(in_features, h_dim, 1, self.dropout)
        self.mlp_r_t = MLP(in_features, h_dim, 1, self.dropout)
        #self.mlp_r_q = MLP(tail_rel, h_dim, 500, self.dropout)

        #self.seq_transformation_r_t = nn.Conv1d(in_features, out_features, kernel_size=1, stride=1, bias=False)
        #self.seq_transformation_s_t = nn.Conv1d(in_features, out_features, kernel_size=1, stride=1, bias=False)

        if self.residual:
            self.proj_residual = nn.Conv1d(in_features, out_features, kernel_size=1, stride=1)

        # self.f_1_h = nn.Conv1d(out_features, 1, kernel_size=1, stride=1)
        # self.f_2_h = nn.Conv1d(out_features, 1, kernel_size=1, stride=1)
        #
        # self.f_1_t = nn.Conv1d(out_features, 1, kernel_size=1, stride=1)
        # self.f_2_t = nn.Conv1d(out_features, 1, kernel_size=1, stride=1)

        self.coef_revise = False
        self.leakyrelu = nn.LeakyReLU()
        self.img_dim = img_dim
        self.h_dim = h_dim
        self.img_proj = nn.Linear(self.img_dim, self.h_dim)

    def get_relation(self, input_h, input_t, relation):#input_r:tensor(185,78)
        num_stock_h = input_h.shape[0]
        # seq_r_h = torch.transpose(input_h, 0, 1).unsqueeze(0)#seq_r:tensor(1,78,185)
        # seq_r_t = torch.transpose(input_t, 0, 1).unsqueeze(0)  # seq_r:tensor(1,78,185)

        logits_1 = torch.zeros(num_stock_h, num_stock_h, device=input_h.device, dtype=input_h.dtype)#logits:tensor(185,185)而且元素全部为0
        #logits_2 = torch.zeros(num_stock_h, num_stock_h, device=input_h.device, dtype=input_h.dtype)
        # seq_fts_r_h = self.seq_transformation_r_h(seq_r_h)#经过卷积层提取特征，seq_fts_r:tensor(1,39,185)
        # seq_fts_r_t = self.seq_transformation_r_t(seq_r_t)
        seq_fts_r_h = self.mlp_r_h(input_h)  # 经过卷积层提取特征，seq_fts_r:tensor(1,39,185)
        seq_fts_r_t = self.mlp_r_t(input_t)

        logits_1 += (torch.transpose(seq_fts_r_h, 1, 0) + seq_fts_r_t).squeeze(0)#logits:tensor(185,185),得到原始特征值
        #coefs = self.mlp_r_q(logits_1)
        coefs = self.img_proj(logits_1)
        coefs = self.leakyrelu(coefs)

        #print(coefs)
        #a = torch.abs(coefs)


        return coefs


    def forward(self, input_h, input_t, relation):
        # unmasked attention
        coefs_eye = self.get_relation(input_h, input_t, relation) #coef_eye:tensor(185,185)
        return coefs_eye