import os
import re
import random
import jieba
import logging
from xml.dom import minidom

import torch
from torchtext.data import Iterator, TabularDataset


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)


def load_dataset(text, label, args, **kwargs):
    train_dataset, dev_dataset, test_dataset = get_dataset('../data', text, label)
    text.build_vocab(train_dataset, dev_dataset, test_dataset)
    label.build_vocab(train_dataset, dev_dataset)
    train_data, dev_data, test_data = Iterator.splits(
        (train_dataset, dev_dataset, test_dataset),
        batch_sizes=(args.batch_size, len(dev_dataset), len(test_dataset)),
        sort_key=lambda x: len(x.text),
        **kwargs)
    return train_data, dev_data, test_data


def word_cut(text):
    jieba.setLogLevel(logging.INFO)
    regex = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9]')
    text = regex.sub(' ', text)
    return [word for word in jieba.cut(text) if word.strip()]


def prepare_csv(path):
    positive = path + '/sample_positive.xml'
    negative = path + '/sample_negative.xml'
    test = path + '/test.xml'
    negative_dom = minidom.parse(negative).documentElement
    positive_dom = minidom.parse(positive).documentElement
    test_dom = minidom.parse(test).documentElement
    positive_lines = positive_dom.getElementsByTagName("review")
    negative_lines = negative_dom.getElementsByTagName("review")
    test_dom_lines = test_dom.getElementsByTagName("review")
    text = []
    for line in positive_lines:
        text.append((line.childNodes[0].data.strip().replace('\n', ' '), 1))
    for line in negative_lines:
        text.append((line.childNodes[0].data.strip().replace('\n', ' '), 0))
    random.shuffle(text)
    train_file = 'train_file.tsv'
    f = open(path + '/' + train_file, 'w', encoding='utf-8')
    f.write('label\ttext\n')
    for i in range(9000):
        f.write(str(text[i][1]) + '\t' + text[i][0] + '\n')
    f = open(path + '/' + 'val_file.tsv', 'w', encoding='utf-8')
    f.write('label\ttext\n')
    for i in range(9000, 10000):
        f.write(str(text[i][1]) + '\t' + text[i][0] + '\n')

    test_file = 'test_file.tsv'
    f = open(path + '/' + test_file, 'w', encoding='utf-8')
    f.write('text\n')
    for line in test_dom_lines:
        f.write(line.childNodes[0].data.strip().replace('\n', ' ') + '\n')


def get_dataset(path, text_field, label_field):
    prepare_csv(path)
    text_field.tokenize = word_cut
    train, dev = TabularDataset.splits(
        path=path, format='tsv', skip_header=True,
        train='train_file.tsv', validation='val_file.tsv',
        fields=[
            ('label', label_field),
            ('text', text_field)
        ]
    )
    test = TabularDataset.splits(
        path=path, format='tsv', skip_header=True,
        train='test_file.tsv',
        fields=[
            ('text', text_field)
        ]
    )[0]
    return train, dev, test
