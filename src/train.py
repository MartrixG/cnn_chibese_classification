import torch
import torch.nn.functional as F

from utils import save


def train(train_iter, dev_iter, model, args):
    if args.cuda:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_acc = 0
    model.train()
    for epoch in range(1, args.epochs + 1):
        steps = 0
        print("\nStarting {:}epoch".format(epoch))
        for batch in train_iter:
            feature, target = batch.text, batch.label
            feature = feature.data.t()
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()
            optimizer.zero_grad()
            res = model(feature)
            loss = F.cross_entropy(res, target)
            loss.backward()
            optimizer.step()
            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(res, 1)[1].view(target.size()).data == target.data).sum()
                train_acc = 100.0 * corrects / batch.batch_size
                print("\r", 'Batch[{}] - loss: {:.6f}  acc: {:.4f}%)'.format(steps, loss.item(), train_acc), end='',
                      flush=True)
            if steps % args.test_interval == 0:
                dev_acc = eval(dev_iter, model, args)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    if args.save_best:
                        print('Saving best model, acc: {:.4f}%'.format(best_acc))
                        save(model, args.save_dir, 'best', '')


def eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature = feature.data.t()
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()
        res = model(feature)
        loss = F.cross_entropy(res, target)
        avg_loss += loss.item()
        corrects += (torch.max(res, 1)[1].view(target.size()).data == target.data).sum()
    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects / size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(avg_loss,
                                                                    accuracy,
                                                                    corrects,
                                                                    size))
    return accuracy


def ceshi(data_iter, model, args):
    model.eval()
    for batch in data_iter:
        feature = batch.text
        feature = feature.data.t()
        if args.cuda:
            feature = feature.cuda()
        res = model(feature)
        res = torch.max(res, 1)[1].data
        return res
