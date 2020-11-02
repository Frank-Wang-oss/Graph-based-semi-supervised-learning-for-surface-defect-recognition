import torch as tr
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time



class distance_func(nn.Module):
    '''
    output a similarity matrix for graph construction
    where
    ratio: can be adjusted according to real applications.
    in_dimen: the dimension of inputs
    '''

    def __init__(self, args, in_dimen, ratio = [2, 2, 1, 1], dropout = False):
        super(distance_func, self).__init__()
        self.args = args
        nf_adj = args.nf_adj

        self.Module_list = []
        for i in range(len(ratio)):
            if i == 0:
                self.Module_list.append(nn.Conv2d(in_dimen, nf_adj*ratio[0], 1))
                if dropout:
                    self.Module_list.append(nn.Dropout(0.6))
            else:
                self.Module_list.append(nn.Conv2d(nf_adj*ratio[i-1], nf_adj*ratio[i], 1))
            self.Module_list.append(nn.BatchNorm2d(nf_adj*ratio[i]))
            self.Module_list.append(nn.LeakyReLU())
        self.Module_list.append(nn.Conv2d(nf_adj*ratio[-1], 1, 1))
        self.Module_list = nn.ModuleList(self.Module_list)

    def forward(self, input):
        '''
        :param input: size is (bs, N, feature_dimen)
        :return: size is (bs, N, N)
        '''
        similarity = self.subtraction(input)
        for l in self.Module_list:
            similarity = l(similarity)
        similarity = tr.transpose(similarity, 1, 3)
        similarity = similarity.squeeze(-1)
        return similarity


    def subtraction(self, input):
        ### input size is (bs, N, feature_dimens)   where N is n_way*n_shot + 1 which needs to be predicted
        A_x = input.unsqueeze(2) # A_x size is (bs, N, 1, feature_dimen)
        A_y = tr.transpose(A_x, 1, 2) # A_y size is (bs, 1, N, feature_dimen)
        subtraction = tr.abs(A_x - A_y) # A_update size is (bs, N, N, feature_dimen)
        subtraction = tr.transpose(subtraction, 1, 3) # A_update size is (bs, feature_dimen, N, N)

        return subtraction


class GCN(nn.Module):
    ### input is [image_embedding, label]
    def __init__(self, args, input_dimens, output_dimens):
        ###
        super(GCN, self).__init__()
        self.args = args
        self.feature_dimen = input_dimens
        self.output  =nn.Linear(self.feature_dimen, output_dimens)

    def forward(self, input_, adj):
        ### adj size is (bs, N, N)
        ### input_ size is (bs, N, feature_dimens)
        u = tr.bmm(adj, input_)
        ### u size is (bs, N, features)
        h = self.output(u)

        return h



class Adj_update(nn.Module):
    '''
    For updating of adjacent matrix
    where
    input_dimens: is the dimension of input

    '''
    def __init__(self, args, input_dimens):
        super(Adj_update, self).__init__()
        self.distance_matrix = distance_func(args, input_dimens)

    def forward(self, adj_before, input):
        '''
        input is used to compute the current state of similarity,
        which then is combined with previous adjacent matrix to
        compute the adjacent matrix in current step

        '''

        W_init = tr.eye(input.size(1)).unsqueeze(0).repeat(input.size(0), 1, 1)
        if tr.cuda.is_available:
            W_init = W_init.cuda()
        distance_matrix = self.distance_matrix(input)
        distance_matrix = distance_matrix - W_init*1e8
        distance_matrix = F.softmax(distance_matrix, 2)
        adj_after = adj_before - distance_matrix

        return adj_after


class GNN(nn.Module):
    '''
    input: the features extracted from CNN
    output: the label of unknown samples

    att:
      nf_gc: the dimensions of hidden layers in graph convolutional network
      input_dimens: the dimensions of input, which are concatenation between embedding and one shot vector
      num_layer: the number of GCN layers

    '''
    def __init__(self, args):
        super(GNN, self).__init__()
        self.args = args
        self.nf_gc = args.nf_gc
        self.input_dimens = args.embedding_size + args.n_way
        self.num_layer = args.num_layer_gc
        self.Adj_update = []
        self.Gnn_update = []
        self.n_way = args.n_way

        for i in range(self.num_layer - 1):
            if i == 0:
                self.Adj_update.append(Adj_update(args,
                                                  self.input_dimens))
                self.Gnn_update.append(GCN(args,
                                           self.input_dimens,
                                           self.nf_gc))
            else:
                self.Adj_update.append(Adj_update(args,
                                                  self.input_dimens + self.nf_gc * (i )))
                self.Gnn_update.append(GCN(args,
                                           self.input_dimens + self.nf_gc * i,
                                           self.nf_gc))

        self.Adj_update = nn.ModuleList(self.Adj_update)
        self.Gnn_update = nn.ModuleList(self.Gnn_update)
        self.Adj_update_final = Adj_update(args,
                                           self.input_dimens + self.nf_gc*(self.num_layer - 1))
        self.Graph_convolution_final = GCN(args,
                                           self.input_dimens + self.nf_gc*(self.num_layer - 1),
                                           self.n_way)

    def forward(self, input):

        ### input size is (bs, N, num_feature) where num_feature is cat(embedding_size, n_way )

        A_init = tr.eye(input.size(1)).unsqueeze(0).repeat(input.size(0), 1, 1)
        if tr.cuda.is_available:
            A_init = A_init.cuda()
        A_update = A_init
        Gc_update = input

        for i in range(self.num_layer - 1):
            A_update = self.Adj_update[i](A_update, Gc_update)

            Gc_update_new = F.leaky_relu(self.Gnn_update[i](Gc_update, A_update)) ### size is (bs, N, num_feature)

            Gc_update = tr.cat([Gc_update_new, Gc_update], 2)
        A_final = self.Adj_update_final(A_update, Gc_update)
        Gn_final = self.Graph_convolution_final(Gc_update, A_final)

        return Gn_final[:,0,:] ### Output the label of the unknown sample



class Embedding(nn.Module):
    ### In this network, samples are extracted as feature vectors

    def __init__(self, args):
        super(Embedding, self).__init__()
        self.emb_size = args.embedding_size
        self.ndf = args.nf_cnn
        self.args = args

        # Input 84x84x3
        self.conv1 = nn.Conv2d(1, self.ndf, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.ndf)

        # Input 42x42x64
        self.conv2 = nn.Conv2d(self.ndf, int(self.ndf * 1.5), kernel_size=3, bias=False)
        self.bn2 = nn.BatchNorm2d(int(self.ndf * 1.5))

        # Input 20x20x96
        self.conv3 = nn.Conv2d(int(self.ndf * 1.5), self.ndf * 2, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.ndf * 2)
        self.drop_3 = nn.Dropout2d(0.4)

        # Input 10x10x128
        self.conv4 = nn.Conv2d(self.ndf * 2, self.ndf * 4, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(self.ndf * 4)
        self.drop_4 = nn.Dropout2d(0.5)

        # Input 5x5x256
        # self.fc1 = nn.Linear(self.ndf * 4 * 5 * 5, self.emb_size, bias=True)
        self.fc1 = nn.Linear(self.ndf * 4 *2*2, self.emb_size, bias=True)
        # self.fc1 = nn.Linear(self.ndf * 4, self.emb_size, bias=True)
        self.bn_fc = nn.BatchNorm1d(self.emb_size)

    def forward(self, input):
        e1 = F.max_pool2d(self.bn1(self.conv1(input)), 2)
        x = F.leaky_relu(e1, 0.2, inplace=True)
        e2 = F.max_pool2d(self.bn2(self.conv2(x)), 2)
        x = F.leaky_relu(e2, 0.2, inplace=True)
        e3 = F.max_pool2d(self.bn3(self.conv3(x)), 2)
        x = F.leaky_relu(e3, 0.2, inplace=True)
        x = self.drop_3(x)
        e4 = F.max_pool2d(self.bn4(self.conv4(x)), 2)
        x = F.leaky_relu(e4, 0.2, inplace=True)
        x = self.drop_4(x)
        x_size = x.size()
        x = x.contiguous()
        x = x.view(x_size[0], -1)
        output = self.bn_fc(self.fc1(x))

        return output


class Embedding_NEU(nn.Module):
    ### In this network, samples are extracted as feature vectors

    def __init__(self, args):
        super(Embedding_NEU, self).__init__()
        self.emb_size = args.embedding_size
        self.ndf = args.nf_cnn
        self.args = args

        # Input 84x84x3
        self.conv1 = nn.Conv2d(1, self.ndf, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.ndf)

        # Input 42x42x64
        self.conv2 = nn.Conv2d(self.ndf, int(self.ndf * 1.5), kernel_size=3, bias=False)
        self.bn2 = nn.BatchNorm2d(int(self.ndf * 1.5))

        # Input 20x20x96
        self.conv3 = nn.Conv2d(int(self.ndf * 1.5), int(self.ndf * 1.5), kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(int(self.ndf * 1.5))

        self.conv4 = nn.Conv2d(int(self.ndf * 1.5), self.ndf * 2, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(self.ndf * 2)
        self.drop_4 = nn.Dropout2d(0.4)

        self.conv5 = nn.Conv2d(self.ndf * 2, self.ndf * 2, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(self.ndf * 2)
        self.drop_5 = nn.Dropout2d(0.4)

        # Input 10x10x128
        self.conv6 = nn.Conv2d(self.ndf * 2, self.ndf * 4, kernel_size=3, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(self.ndf * 4)
        self.drop_6 = nn.Dropout2d(0.5)

        # Input 5x5x256
        # self.fc1 = nn.Linear(self.ndf * 4 * 5 * 5, self.emb_size, bias=True)
        self.fc1 = nn.Linear(self.ndf * 4 , self.emb_size, bias=True)
        self.bn_fc = nn.BatchNorm1d(self.emb_size)

    def forward(self, input):
        e1 = F.max_pool2d(self.bn1(self.conv1(input)), 2)
        x = F.leaky_relu(e1, 0.2, inplace=True)
        e2 = F.max_pool2d(self.bn2(self.conv2(x)), 2)
        x = F.leaky_relu(e2, 0.2, inplace=True)
        e3 = F.max_pool2d(self.bn3(self.conv3(x)), 2)
        x = F.leaky_relu(e3, 0.2, inplace=True)
        e4 = F.max_pool2d(self.bn4(self.conv4(x)),2)
        x = F.leaky_relu(e4, 0.2, inplace=True)
        x = self.drop_4(x)
        e5 = self.bn5(self.conv5(x))
        x = F.leaky_relu(e5, 0.2, inplace=True)
        x = self.drop_5(x)
        e6 = self.bn6(self.conv6(x))
        x = F.leaky_relu(e6, 0.2, inplace=True)
        x = self.drop_6(x)


        x_size = x.size()
        x = x.contiguous()
        x = x.view(x_size[0], -1)
        output = self.bn_fc(self.fc1(x))

        return output
