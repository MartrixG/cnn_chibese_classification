import argparse
import torch
import torchtext.data as data

import model
import train
from utils import load_dataset

parser = argparse.ArgumentParser(description='TextCNN text classifier')
# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=5, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=32, help='batch size for training [default: 128]')
parser.add_argument('-log-interval', type=int, default=1,
                    help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=10,
                    help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-dir', type=str, default='log/', help='where to save the snapshot')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# model
parser.add_argument('-dropout', type=float, default=0.2, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embedding-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('-filter-num', type=int, default=100, help='number of each size of filter')
parser.add_argument('-filter-sizes', type=str, default='3,4,5',
                    help='comma-separated filter sizes to use for convolution')

# device
parser.add_argument('-device', type=int, default=0, help='device to use for iterate data, -1 mean cpu [default: -1]')

# option
parser.add_argument('-snapshot', type=str, default='log/best_.pt', help='filename of model snapshot [default: None]')
parser.add_argument('-train', type=str, default='test')
args = parser.parse_args()


if __name__ == "__main__":
    print('Loading data...')
    text_field = data.Field(lower=True)
    label_field = data.Field(sequential=False, use_vocab=False, unk_token=None)
    train_iter, dev_iter, test_iter = load_dataset(text_field, label_field, args, device=-1, repeat=False, shuffle=True)

    args.vocabulary_size = len(text_field.vocab)
    args.class_num = len(label_field.vocab)
    args.cuda = args.device != -1 and torch.cuda.is_available()
    args.filter_sizes = [int(size) for size in args.filter_sizes.split(',')]

    print('Parameters:')
    for attr, value in sorted(args.__dict__.items()):
        if attr in {'vectors'}:
            continue
        print('\t{}={}'.format(attr.upper(), value))

    text_cnn = model.TextCNN(args)
    if args.train != 'train':
        print('\nLoading model from {}...\n'.format(args.snapshot))
        text_cnn.load_state_dict(torch.load(args.snapshot))
        if args.cuda:
            text_cnn = text_cnn.cuda()
        ans = train.ceshi(test_iter, text_cnn, args)
        f = open('log/ans.csv', 'w')
        count = 0
        for i in ans:
            count += 1
            f.write(str(count) + ',' + str(i.item()))
            f.write('\n')
        exit(0)

    if args.cuda:
        torch.cuda.set_device(args.device)
        text_cnn = text_cnn.cuda()
    try:
        train.train(train_iter, dev_iter, text_cnn, args)
    except KeyboardInterrupt:
        print('Exiting from training early')
