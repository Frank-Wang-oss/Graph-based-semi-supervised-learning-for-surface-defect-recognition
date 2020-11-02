import torch as tr
from PIL import Image,ImageOps
import torchvision as tv
import numpy as np
import os
import random


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tr.manual_seed(seed)
    tr.cuda.manual_seed(seed)
    tr.backends.cudnn.deterministic = True


class data_loader():
    def __init__(self, args):
        self.args = args
        self.image_size = args.Image_size
        path = './NEU-CLS/'
        self.label_e = args.num_labeled
        self.unlabel_e = args.num_unlabeled
        self.trans = tv.transforms.Compose([
            tv.transforms.ToTensor()
            ])
        self.trans_label = tv.transforms.Compose([
            tv.transforms.Normalize([0.5034],
                                    [0.2088])
            ])
        self.trans_unlabel = tv.transforms.Compose([
            tv.transforms.Normalize([0.5004],
                                    [0.2099])
            ])
        self.trans_test = tv.transforms.Compose([
            tv.transforms.Normalize([0.5138],
                                    [0.2145])
            ])

        self.train_labeled, self.train_unlabeled, self.test, \
        self.test_label = self.read(path)

    def read(self, path):
        dict_cls = {
            0: 'Cr',
            1: 'ln',
            2: 'Pa',
            3: 'PS',
            4: 'RS',
            5: 'Sc'
            }

        label_encoder = {}
        train_label = {}
        train_unlabel = []
        test = []
        test_label = []
        np.random.seed(0)
        image = {}
        ### Read file name from directory
        for k,v in dict_cls.items():
            label_encoder[v] = k
        for i,class_name in enumerate(os.listdir(path)):
            image[class_name] = []
            for file_name in os.listdir(path+class_name):
                image[class_name].append(file_name)

        for i in image.keys():
            train = []
            name = image[i]
            np.random.shuffle(name)

            ### Split name into train_label, train_unlabel and test dataset
            train_label_name = name[:self.label_e]
            train_unlabel_name = name[self.label_e:(self.label_e+self.unlabel_e)]
            test_name = name[(self.label_e+self.unlabel_e):]

            ### Read training samples with labels from directory
            for j in train_label_name:
                img = self.image_process(path + i +'/' + j)
                img = self.trans_label(img)
                train.append(img)
            train = tr.stack(train,0)
            train_label[label_encoder[i]] = train

            ### Read training samples without labels from directory
            for j in train_unlabel_name:
                img = self.image_process(path + i +'/' + j)
                img = self.trans_unlabel(img)
                train_unlabel.append(img)

            ### Read testing samples from directory
            for j in test_name:
                img = self.image_process(path + i + '/' + j)
                img = self.trans_test(img)
                test.append(img)
                test_label.append(tr.LongTensor([label_encoder[i]]))

        np.random.shuffle(train_unlabel)
        if len(train_unlabel) != 0:
            train_unlabel = tr.stack(train_unlabel, 0)
        else:
            train_unlabel = tr.Tensor()
        test = tr.stack(test, 0)
        test_label = tr.cat((test_label),0)

        return train_label, train_unlabel, test, test_label



    def image_process(self, path):
        img = Image.open(path)
        img = img.resize((self.image_size, self.image_size), Image.ANTIALIAS)
        img = self.trans(img)

        return img

    def compute_mean_std(self, dict_):
        assemble = []
        mean = []
        std = []
        for k,v in dict_.items():
            assemble.append(v)
        assemble = tr.cat(assemble,0)
        channel = assemble.size(1)
        for i in range(channel):
            mean.append(tr.mean(assemble[:,i,:,:]))
            std.append(tr.std(assemble[:,i,:,:]))

        return mean, std


def trans_cuda(input):
    if tr.cuda.is_available():
        input = input.cuda()
    return input

def graph_train(args, train):
    ### For the graph construction when training
    ### train must be a dict.
    Nn = args.n_shot
    Nc = args.n_way
    xi = []
    xi_label = []

    xs = []
    xs_label = []
    xl_onehot = []


    for bs in range(args.batch_size):
        ### sample training samples in one batch size
        xs_i = []
        xs_label_i = []
        xl_onehot_i = []
        random_cls = random.sample(range(Nc),1)
        for cls in range(Nc):
            data_class = train[cls]
            num_cls = data_class.size(0)
            ### Sample an unknown sample
            if cls == random_cls[0]:
                index = random.sample(range(num_cls), Nn+1)
                xi.append(data_class[index[0]])
                xi_label.append(tr.LongTensor([cls]))
                xs_i.extend(data_class[index[1:]])
            ### Sample Nn samples from each class
            else:
                index = random.sample(range(num_cls), Nn)
                xs_i.extend(data_class[index])

            xs_label_i.append(tr.LongTensor([cls]).repeat(Nn))
            onehot = tr.zeros(Nc)
            onehot[cls] = 1
            xl_onehot_i.append(onehot.repeat(Nn,1))

        index = np.random.permutation(np.arange(Nc * Nn))
        xs.append(tr.stack(xs_i, 0)[index])
        xs_label.append(tr.cat(xs_label_i, 0)[index])
        xl_onehot.append(tr.cat(xl_onehot_i, 0)[index])
    xi = tr.stack(xi, 0)
    xi_label = tr.cat(xi_label, 0)
    xs = tr.stack(xs,0)
    xs_label = tr.stack(xs_label, 0)
    xl_onehot = tr.stack(xl_onehot, 0)

    return xi, xi_label, xs, xs_label, xl_onehot


def graph_prediction(args, train, prediction):
    ### For the graph construction when predicting
    Nn = args.n_shot
    Nc = args.n_way
    xi = []
    xs = []
    xs_label = []
    xl_onehot = []

    for bs in range(args.batch_size):
        ### sample train data
        xs_i = []
        xs_label_i = []
        xl_onehot_i = []
        for cls in range(Nc):
            data_class = train[cls]
            num_cls = data_class.size(0)
            index = random.sample(range(num_cls), Nn)
            xs_i.extend(data_class[index])
            xs_label_i.append(tr.LongTensor([cls]).repeat(Nn))
            onehot = tr.zeros(Nc)
            onehot[cls] = 1
            xl_onehot_i.append(onehot.repeat(Nn, 1))

        index = np.random.permutation(np.arange(Nc * Nn))
        xs.append(tr.stack(xs_i, 0)[index])
        xs_label.append(tr.cat(xs_label_i, 0)[index])
        xl_onehot.append(tr.cat(xl_onehot_i, 0)[index])
        xi.append(prediction[bs])

    xi = tr.stack(xi, 0)
    xs = tr.stack(xs, 0)
    xs_label = tr.stack(xs_label, 0)
    xl_onehot = tr.stack(xl_onehot, 0)



    return xi, xs, xs_label, xl_onehot

