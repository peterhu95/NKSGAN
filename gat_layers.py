# coding=utf-8

import numpy as np
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
from torch.autograd import Variable
from copy import deepcopy
import datetime
import sys

CUDA = torch.cuda.is_available()


class ConvKB(nn.Module):
    def __init__(self, input_dim, input_seq_len, in_channels, out_channels, drop_prob, alpha_leaky):
        super().__init__()

        self.conv_layer = nn.Conv2d(
            in_channels, out_channels, (1, input_seq_len))  # kernel size -> 1*input_seq_length(i.e. 2)
        self.dropout = nn.Dropout(drop_prob)
        self.non_linearity = nn.ReLU()
        self.fc_layer = nn.Linear((input_dim) * out_channels, 1)

        nn.init.xavier_uniform_(self.fc_layer.weight, gain=1.414)
        nn.init.xavier_uniform_(self.conv_layer.weight, gain=1.414)

    def forward(self, conv_input):

        batch_size, length, dim = conv_input.size()
        # assuming inputs are of the form ->
        conv_input = conv_input.transpose(1, 2)
        # batch * length(which is 3 here -> entity,relation,entity) * dim
        # To make tensor of size 4, where second dim is for input channels
        conv_input = conv_input.unsqueeze(1)

        out_conv = self.dropout(
            self.non_linearity(self.conv_layer(conv_input)))

        input_fc = out_conv.squeeze(-1).view(batch_size, -1)
        output = self.fc_layer(input_fc)
        return output


class SpecialSpmmFunctionFinal(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, edge, edge_w, N, E, out_features):  # tar_bad == None out_features=1, calculating a
        
        # 这里还没处理好edge_w的三元组表示的头尾实体的顺序问题，edge_w都是尾实体在前头实体在中间，可否考虑在生成edge_w时就生成两个不同的edge_w，一个是edge_w1尾实体表示在前，一个是edge_w2头实体表示在前。同时也要有两个edge，一个edge，一个edge_over
        # 还有一个问题是，这里涉及到了tar_bad即久程序的编号，和edge即新程序的编号，检查一下两者是否统一？
        
        a = torch.sparse_coo_tensor(
            edge, edge_w, torch.Size([N, N, out_features]))  # 以edge为索引，以edge_w为值，值的数量为 2*训练数据三元组个数 ，稀疏张量大小为(N, N, out_features)
        # 其实就是构建特殊的(因为值不是关系也不是1，而是信息)稀疏邻接张量，例子：实体1和2直接通过关系1相连，则他们的id分别作为列和行索引。attention层算出的1跳信息，则作为稀疏张量的值。
        # 即实体2的id为第一维，实体1的id为第二维，第三维默认都为1
        # 最后，a的大小(实体总的数目, 实体总的数目, 1)，直观理解：0维表示尾实体，1维表示头实体，2维表示这个尾实体和某个头实体的attention信息
        b = torch.sparse.sum(a, dim=1)  # 按1维，求和。得出大小(实体总的数目, 1)的矩阵。表示每个实体的综合的邻居信息。
        ctx.N = b.shape[0]        # b的第一个维度
        ctx.outfeat = b.shape[1]  # b的第二个维度
        ctx.E = E  # E是2*训练数据三元组个数
        ctx.indices = a._indices()[0, :]  # a的索引，即edge。0行，即所有邻居节点的实体id（1跳实体id 和 2跳实体id）

        return b.to_dense()  # 返回实体的邻居信息张量(实体总的数目, 1)
            

    @staticmethod
    def backward(ctx, grad_output):
        grad_values = None
        if ctx.needs_input_grad[1]:
            edge_sources = ctx.indices

            if(CUDA):
                edge_sources = edge_sources.cuda()

            grad_values = grad_output[edge_sources]
            # grad_values = grad_values.view(ctx.E, ctx.outfeat)
            # print("Grad Outputs-> ", grad_output)
            # print("Grad values-> ", grad_values)
        return None, grad_values, None, None, None


class SpecialSpmmFinal(nn.Module):
    def forward(self, edge, edge_w, N, E, out_features):
        return SpecialSpmmFunctionFinal.apply(edge, edge_w, N, E, out_features)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, num_nodes, in_features, out_features, nrela_dim, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_nodes = num_nodes
        self.alpha = alpha
        self.concat = concat
        self.nrela_dim = nrela_dim

        self.a = nn.Parameter(torch.zeros((out_features, 2 * in_features + nrela_dim)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)
        self.a_2 = nn.Parameter(torch.zeros((1, out_features)))
        nn.init.xavier_normal_(self.a_2.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm_final = SpecialSpmmFinal()

    def forward(self, input, edge, edge_embed): # , edge_list_nhop, edge_embed_nhop
        N = input.size()[0]
        #print("a:\n",self.a)
        #print("a2:\n",self.a_2)

        # Self-attention on the nodes - Shared attention mechanism
        edge = edge[:, :].long()

        edge_h = torch.cat(
            (input[edge[0, :], :], input[edge[1, :], :], edge_embed[:, :]), dim=1).t()
        #edge_embed.cpu()
        # edge_h: (2*in_dim + nrela_dim) x E

        #edge_h=Variable(edge_h)
        edge_m = self.a.cuda().mm(edge_h)
        # edge_m: D * E

        # to be checked later
        powers = -self.leakyrelu(self.a_2.cuda().mm(edge_m).squeeze())
        #print("a:",self.a)
        #print("edge_h:",edge_h)
        #print("a_2:",self.a_2)
        #print("edge_m:",edge_m)
        #print("powers:",powers)
        edge_e = torch.exp(powers).unsqueeze(1)
        #assert (np.isnan(edge_e.cpu().data.numpy()) == False).all()  # all elements must be not nan!  all not nan => condiction True
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm_final(
            edge, edge_e, N, edge_e.data.shape[0], 1)
        e_rowsum[e_rowsum == 0.0] = 1e-12

        e_rowsum = e_rowsum
        # e_rowsum: N x 1
        edge_e = edge_e.squeeze(1)

        edge_e = self.dropout(edge_e)
        # edge_e: E

        edge_w = (edge_e * edge_m).t()
        # edge_w: E * D
        
        h_prime = self.special_spmm_final(
            edge, edge_w, N, edge_w.data.shape[0], self.out_features)

        #assert (np.isnan(h_prime.cpu().data.numpy()) == False).all()
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out

        #assert (np.isnan(h_prime.cpu().data.numpy()) == False).all()
        assert not torch.isnan(h_prime).any()
        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
