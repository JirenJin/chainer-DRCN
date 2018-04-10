import argparse
import pprint

import matplotlib
matplotlib.use('agg')

import chainer
import chainer.functions as F
from chainer.iterators import SerialIterator
from chainer.training import Trainer
from chainer.training import extensions
from chainer.training.triggers import MaxValueTrigger

import dataset
import drcn
from updater import Updater
import utils


class LossAndAccuracy(chainer.Chain):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def __call__(self, x, t):
        logits = self.model.classify(self.model.encode(x))
        loss = F.softmax_cross_entropy(logits, t)
        acc = F.accuracy(logits, t)
        chainer.report({'loss_cla_t': loss})
        chainer.report({'acc_t': acc})
        return loss


def main(args):
    s_train, s_test = dataset.load_svhn()
    t_train, t_test = dataset.load_mnist()
    s_train_iter = SerialIterator(
        s_train, args.batchsize, shuffle=True, repeat=True)
    t_train_iter = SerialIterator(
        t_train, args.batchsize, shuffle=True, repeat=True)
    s_test_iter = SerialIterator(
        s_test, args.batchsize, shuffle=False, repeat=False)
    t_test_iter = SerialIterator(
        t_test, args.batchsize, shuffle=False, repeat=False)

    model = drcn.DRCN()
    target_model = LossAndAccuracy(model)
    loss_list = ['loss_cla_s', 'loss_cla_t', 'loss_rec']
    optimizer = chainer.optimizers.Adam(args.lr)
    optimizer.setup(model)
    optimizers = {
        'model': optimizer
    }

    updater = Updater(s_train_iter, t_train_iter, optimizers, args)
    out_dir = utils.prepare_dir(args)
    trainer = Trainer(updater, (args.max_iter, 'iteration'), out=out_dir)
    trainer.extend(extensions.LogReport(trigger=(args.interval, args.unit)))
    trainer.extend(
        extensions.snapshot_object(model, filename='model'),
        trigger=MaxValueTrigger('acc_t', (args.interval, args.unit)))
    trainer.extend(extensions.Evaluator(t_test_iter, target_model,
                                        device=args.device), trigger=(args.interval, args.unit))
    trainer.extend(extensions.PrintReport([args.unit, *loss_list, 'acc_s', 'acc_t', 'elapsed_time']))
    trainer.extend(extensions.PlotReport([*loss_list], x_key=args.unit, file_name='loss.png', trigger=(args.interval, args.unit)))
    trainer.extend(extensions.PlotReport(['acc_s', 'acc_t'], x_key=args.unit, file_name='accuracy.png', trigger=(args.interval, args.unit)))
    trainer.extend(extensions.ProgressBar(update_interval=1))
    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", "-g", type=int, default=0)
    parser.add_argument("--max_iter", "-i", type=int, default=100)
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--batchsize", "-b", type=int, default=128)
    parser.add_argument("--lr", "-lr", type=float, default=3e-4)
    parser.add_argument("--out", type=str, default='result')
    parser.add_argument("--unit", type=str, choices=['iteration', 'epoch'], default="iteration")
    parser.add_argument("--noise", type=str, choices=['no_noise', 'impulse', 'gaussian'], default="impulse")
    parser.add_argument("--source_only", type=int, default=0)
    args = parser.parse_args()
    pprint.pprint(vars(args))

    main(args)
