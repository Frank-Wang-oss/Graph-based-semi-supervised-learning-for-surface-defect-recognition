import argparse as arg


def args():
    args = arg.ArgumentParser()

    args.add_argument('--nf_adj', default=96, type = int, help='the hidden number of elements in adjacency matrix')
    args.add_argument('--nf_gc', default= 64, type = int, help = 'the number of elements in hidden layer of Graph convolution')
    args.add_argument('--nf_cnn', default=64, type = int, help = 'the number of elements in hidden layer of image embedding')
    args.add_argument('--lr',default=3e-4,type = int, help = 'learning rate, alternative choices are 3e-4,1e-3')
    args.add_argument('--num_layer_gc', default=5, type = int, help = 'the numebr of layers in graph convolution')
    args.add_argument('--n_way', default=6, type = int, help = 'the number of classes choosed')
    args.add_argument('--n_shot', default= 3, type = int, help = 'the number of samples from each class')
    args.add_argument('--batch_size', default=18, type = int, help = 'the number of batch size')
    args.add_argument('--embedding_size', default=96, type = int, help = 'embedding size')
    args.add_argument('--seed', default=1, type = int, help = 'random seed')
    args.add_argument('--iteration', default=6001, type = int)
    args.add_argument('--interval', default= 100, type = int, help = 'interval for printing loss')
    args.add_argument('--sample_eval', default= 1400, type = int)
    args.add_argument('--save_name', default='Test', type = str)
    args.add_argument('--Image_size', default=48, type = int)
    args.add_argument('--eval_ratio', default=98.5, type = float, help = 'threshold for next training')
    args.add_argument('--num_cycle', default=9, type = int, help = 'the number of training cycles for semi-learning')
    args.add_argument('--pre_num', default=20,type = int, help = 'the number of micrograph for prediction')
    args.add_argument('--add_interval', default=300, type = int)
    args.add_argument('--num_labeled',default=50,type = int)
    args.add_argument('--num_unlabeled', default=200,type = int)

    return args.parse_args()


