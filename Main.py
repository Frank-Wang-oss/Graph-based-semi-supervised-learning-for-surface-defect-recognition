import torch as tr
import numpy as np
import torch.optim as opt
from Data_Loader import data_loader, graph_train, graph_prediction
from Argument import args
import Model
import torch.nn as nn
import time
import random
import os
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class main():
    def __init__(self, args):
        self.args = args
        data = data_loader(args)
        self.train_labeled = data.train_labeled
        self.train_unlabeled = data.train_unlabeled
        self.test = data.test
        self.initial_num = self.train_labeled[0].size(0)
        if tr.cuda.is_available():
            self.train_unlabeled = self.train_unlabeled.cuda()
            self.test = self.test.cuda()
            self.test_label = data.test_label.cuda()
        self.lr = args.lr
        self.dec_lr = 1500

        self.embed_module = Model.Embedding(args)
        self.GNN_module = Model.GNN(args)
        if tr.cuda.is_available():
            self.embed_module = self.embed_module.cuda()
            self.GNN_module = self.GNN_module.cuda()
        self.opt_embed = opt.Adam(self.embed_module.parameters(),
                                  lr = self.lr,
                                  weight_decay=1e-6)
        self.opt_gnn = opt.Adam(self.GNN_module.parameters(),
                                lr = self.lr,
                                weight_decay=1e-6)
        self.loss = nn.CrossEntropyLoss()
        self.add_count = 0
        self.bs = args.batch_size

    def Train_batch(self, iteration):
        ### For training
        self.embed_module.train()
        self.GNN_module.train()

        data = graph_train(self.args, self.train_labeled)
        if tr.cuda.is_available():
            data_cuda = [data_.cuda() for data_ in data]
            data = data_cuda

        self.opt_embed.zero_grad()
        self.opt_gnn.zero_grad()

        ### Extract feature vectors from Nc*Nn + 1 samples
        xi, xi_label, xs, xs_label, xl_onehot = data
        xi_embed = self.embed_module(xi)
        xs_embed_ = [self.embed_module(xs[:,i,:,:,:]) for i in range(xs.size(1))]
        xs_embed = tr.stack(xs_embed_, 1)
        x_features = tr.cat((xi_embed.unsqueeze(1),xs_embed), 1)
        uniform_pad = tr.Tensor(xl_onehot.size(0), 1, xl_onehot.size(2)).fill_(1.0 / xl_onehot.size(2)).cuda()
        x_label = tr.cat([uniform_pad, xl_onehot], 1)
        nodes_feature = tr.cat([x_features, x_label], 2)


        out = self.GNN_module(nodes_feature)
        loss = self.loss(out, xi_label)

        loss.backward()

        self.opt_embed.step()
        self.opt_gnn.step()

        self.adjust_learning_rate(optimizers=[self.opt_embed,self.opt_gnn], lr = self.lr, iter=iteration)

        return loss.item()


    def prediction(self, unlabel_input):
        ### For predicting unlabeled samples and testing
        self.embed_module.eval()
        self.GNN_module.eval()

        data = graph_prediction(self.args, self.train_labeled, unlabel_input)
        if tr.cuda.is_available():
            data_cuda = [data_.cuda() for data_ in data]
            data = data_cuda

        [xi, xs, xs_label, xl_onehot] = data

        xi_embed = self.embed_module(xi)
        xs_embed_ = [self.embed_module(xs[:, i, :, :, :]) for i in range(xs.size(1))]
        xs_embed = tr.stack(xs_embed_, 1)
        x_features = tr.cat((xi_embed.unsqueeze(1), xs_embed), 1)
        uniform_pad = tr.Tensor(xl_onehot.size(0), 1, xl_onehot.size(2)).fill_(1.0 / xl_onehot.size(2)).cuda()
        x_label = tr.cat([uniform_pad, xl_onehot], 1)
        nodes_feature = tr.cat([x_features, x_label], 2)


        out = self.GNN_module(nodes_feature)

        return out

    def eval_train(self):
        self.embed_module.eval()
        self.GNN_module.eval()

        correct = 0
        total = 0
        sample_eval = self.args.sample_eval
        iteration = int(sample_eval/self.args.batch_size)
        for i in range(iteration):

            data = graph_train(self.args, self.train_labeled)
            if tr.cuda.is_available():
                data_cuda = [data_.cuda() for data_ in data]
                data = data_cuda

            [xi, xi_label, xs, xs_label, xl_onehot] = data
            xi_embed = self.embed_module(xi)
            xs_embed_ = [self.embed_module(xs[:, i, :, :, :]) for i in range(xs.size(1))]
            xs_embed = tr.stack(xs_embed_, 1)
            x_features = tr.cat((xi_embed.unsqueeze(1), xs_embed), 1)
            uniform_pad = self.trans_cuda(tr.Tensor(xl_onehot.size(0), 1, xl_onehot.size(2)).fill_(1.0/xl_onehot.size(2)))
            x_label = tr.cat([uniform_pad, xl_onehot], 1)
            nodes_feature = tr.cat([x_features, x_label], 2)


            out = self.GNN_module(nodes_feature)
            out = tr.argmax(out, 1)
            for j in range(out.size(0)):
                total = total + 1
                if out[j] == xi_label[j]:
                    correct = correct + 1

        accu = (correct/total)*100


        return accu, correct, total


    def semi(self):
        train_loss_ = []
        test_loss_ = []
        train_accu_ = []
        test_accu_ = []
        confusion_m_ = []
        time0 = time.time()
        wrong_name_ = []
        count = 0
        num = int((self.train_unlabeled.size(0))/self.args.num_cycle)
        loss_comp = 0
        loss_count = 0

        for i in range(self.args.iteration):
            loss = self.Train_batch(i)
            loss_comp = loss_comp + loss
            loss_count = loss_count + 1

            if i% self.args.interval == 0:
                self.add_count = self.add_count + 1
                loss = loss_comp/loss_count
                loss_comp = 0
                loss_count = 0
                train_accu, correct_train, total_train = self.eval_train()
                if self.add_count >=int(self.args.add_interval/self.args.interval):
                    self.add_count = 0

                    if train_accu >= self.args.eval_ratio:
                        ### If training accuracy > threshold, training pauses and labeling starts
                        if count*num<self.train_unlabeled.size(0):
                            prediction_data_set = self.train_unlabeled[count*num:(count+1)*num]
                            pre_ite = int(prediction_data_set.size(0)/self.args.batch_size)
                            extra = prediction_data_set.size(0) - pre_ite*self.args.batch_size
                            ### Output the results in every micrographs
                            for bs in range(pre_ite):
                                mode = []
                                prediction_data = prediction_data_set[bs*self.args.batch_size:(bs+1)*self.args.batch_size]
                                for _ in range(self.args.pre_num):
                                    out = self.prediction(prediction_data)
                                    out = tr.argmax(out, 1)
                                    mode.append(out)
                                ### Vote the results
                                mode_out = self.mode(mode,1)
                                ### Adding newly labeled datasets into labeled dataset
                                for j in range(out.size(0)):
                                    pre_cls = int(mode_out[j])
                                    pre_data = prediction_data[j].unsqueeze(0).cpu()
                                    self.train_labeled[pre_cls] = tr.cat((pre_data, self.train_labeled[pre_cls]),0)

                            if extra != 0:
                                mode = []
                                prediction_data = prediction_data_set[pre_ite*self.args.batch_size:]
                                self.args.batch_size = extra
                                for _ in range(self.args.pre_num):
                                    out = self.prediction(prediction_data)
                                    out = tr.argmax(out, 1)
                                    mode.append(out)
                                mode_out = self.mode(mode, 1)
                                for j in range(out.size(0)):
                                    pre_cls = int(mode_out[j])
                                    pre_data = prediction_data[j].unsqueeze(0).cpu()
                                    self.train_labeled[pre_cls] = tr.cat((pre_data, self.train_labeled[pre_cls]), 0)
                                self.args.batch_size = self.bs
                            count = count + 1

                test_accu,total,correct,loss_test, confusion_m,wrong_name = self.test_eval()
                time_interval = time.time() - time0
                added_num = 0
                for m in range(6):
                    added_num = added_num + (self.train_labeled[m].size(0)-self.initial_num)
                print('------------The {}th iteration--------------'.format(i))
                print('count is {}'.format(count))
                print('added number is {}'.format(added_num))
                print('Training loss is {}'.format(loss))
                print('Testing loss is {}'.format(loss_test))
                print('Training accuracy is {}, correct/total is {}/{}'.format(train_accu,correct_train,total_train))
                print('Testing accuracy is {}, correct/total is {}/{}'.format(test_accu,correct, total))
                print('Cost is {}'.format(time_interval))
                test_loss_.append(loss_test)
                train_loss_.append(loss)
                train_accu_.append(train_accu)
                test_accu_.append(test_accu)
                confusion_m_.append(confusion_m)
                wrong_name_.append(wrong_name)
                time0 = time.time()
        final_test_accu = self.test_eval()
        print('------------Final testing accuracy is {}'.format(final_test_accu))
        np.save('./experiment_data/{}.npy'.format(self.args.save_name),
                [test_loss_,train_loss_,train_accu_,test_accu_,confusion_m_,wrong_name_])
        tr.save(self.embed_module.state_dict(),'./checkpoint/embed_para_{}.pth'.format(self.args.save_name))
        tr.save(self.GNN_module.state_dict(), './checkpoint/GNN_para_{}.pth'.format(self.args.save_name))

    def test_eval(self):
        self.embed_module.eval()
        self.GNN_module.eval()
        real = []
        prediction = []
        loss = 0
        total = 0
        correct = 0
        count = 0
        num = self.test.size(0)
        bs = self.args.batch_size
        iteration = int(num/bs)
        wrong_name = []
        for i in range(iteration):
            prediction_data = self.test[count*bs:(count+1)*bs]
            label_data = self.test_label[count*bs:(count+1)*bs]
            label_data_name = self.test_name[count*bs:(count+1)*bs]

            out = self.prediction(prediction_data)
            loss_i = self.loss(out,label_data)
            loss = loss + loss_i.item()
            mode = []
            ### Output the results of micrographs and vote
            for _ in range(self.args.pre_num):
                out = self.prediction(prediction_data)
                out = tr.argmax(out, 1)
                mode.append(out)
            out = self.mode(mode, 1)

            for j in range(out.size(0)):
                total = total + 1
                if out[j] == label_data[j]:
                    correct = correct + 1
                elif out[j] != label_data[j]:
                    wrong_name.append(label_data_name[j])
            count = count+1
            real.append(label_data.cpu().numpy())
            prediction.append(out.cpu().numpy())
        real = np.concatenate(real,0)
        prediction = np.concatenate(prediction, 0)
        wrong_name = np.stack(wrong_name,0)
        loss = loss/iteration
        accu = (correct/total)*100
        confusion_m = confusion_matrix(real, prediction)
        return accu, total, correct, loss, confusion_m,wrong_name

    def trans_cuda(self, input):
        if tr.cuda.is_available():
            input = input.cuda()
        return input

    def mode(self, list, dim):
        out_mode = []
        stack_list = tr.stack(list, dim)
        for i in range(stack_list.size(0)):
            sample_i = stack_list[i]
            mode = tr.bincount(sample_i)
            mode = tr.argmax(mode)
            out_mode.append(mode)
        out_mode = tr.stack(out_mode, 0)

        return out_mode

    def visualization(self, data, label, name):
        ###data is (bs, N, feature_dimension), and torch format
        ###label is (bs, N), and torch format
        data_single = data[0].detach().cpu().numpy()
        label_single = label[0].detach().cpu().numpy()

        tsne = TSNE()
        reduce_dimension = tsne.fit_transform(data_single)

        plt.scatter(reduce_dimension[:, 0], reduce_dimension[:, 1], c=[6]+label_single)
        plt.show()
        plt.savefig('./experiment_data/{}_{}.png'.format(self.args.save_name,name))
        plt.close()

    def adjust_learning_rate(self, optimizers, lr, iter):
        new_lr = lr * (0.5 ** (int(iter / self.dec_lr)))

        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

if __name__ == '__main__':
    args = args()
    args.embedding_size = 64
    args.nf_adj = 64
    main_ = main(args)
    main_.semi()