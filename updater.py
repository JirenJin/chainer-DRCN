import copy

import numpy as np
import chainer
from chainer import cuda
import chainer.functions as F

from base_da_updater import BaseDAUpdater


def get_impulse_noise(X, level):
    p = 1. - level
    device = cuda.get_device_from_array(X.data)
    X = cuda.to_cpu(X)
    Y = X * np.random.binomial(1, p, size=X.shape).astype('f')
    Y = cuda.to_gpu(Y, device=device)
    return Y


def get_gaussian_noise(X, std):
    device = cuda.get_device_from_array(X.data)
    X = cuda.to_cpu(X)
    Y = np.random.normal(X, scale=std).astype('f')
    Y = np.clip(Y, 0., 1.)
    Y = cuda.to_gpu(Y, device=device)
    return Y


class Updater(BaseDAUpdater):
    """DRCN Updater."""
    def __init__(self, s_iter, t_iter, optimizers, args):
        super().__init__(s_iter, t_iter, optimizers, device=args.device)
        self.model = optimizers['model'].target
        self.noise = args.noise
        self.source_only = args.source_only

    def update_core(self):
        cuda.Device(self.device).use()
        xp = self.model.xp
        if not self.source_only:
            # autoencoder training
            loss_rec_data = 0
            n_batch = 0
            total_batches = len(self.t_iter.dataset) / self.t_iter.batch_size
            for t_batch in self.t_iter:
                t_imgs, _ = self.converter(t_batch, self.device)
                t_imgs_copy = copy.deepcopy(t_imgs)
                # whether to use denoising autoencoder
                if self.noise == 'impulse':
                    t_imgs = get_impulse_noise(t_imgs, 0.5)
                elif self.noise == 'gaussian':
                    t_imgs = get_gaussian_noise(t_imgs, 0.5)
                elif self.noise == 'no_noise':
                    pass
                else:
                    raise NotImplementedError
                t_encoding = self.model.encode(t_imgs)
                t_decoding = self.model.decode(t_encoding)
                loss_rec = F.mean_squared_error(t_decoding, t_imgs_copy)
                for opt in self.optimizers.values():
                    opt.target.cleargrads()
                loss_rec.backward()
                for opt in self.optimizers.values():
                    opt.update()
                loss_rec_data += loss_rec.data
                n_batch += 1
                if n_batch >= total_batches:
                    break
            loss_rec_data /= n_batch

        # encoder and classifier training
        loss_cla_s_data = 0
        acc_s_data = 0
        n_batch = 0
        total_batches = len(self.s_iter.dataset) / self.s_iter.batch_size
        for s_batch in self.s_iter:
            s_imgs, s_labels = self.converter(s_batch, self.device)
            s_encoding = self.model.encode(s_imgs)
            s_logits = self.model.classify(s_encoding)
            loss_cla_s = F.softmax_cross_entropy(s_logits, s_labels)
            acc_s = F.accuracy(s_logits, s_labels)
            for opt in self.optimizers.values():
                opt.target.cleargrads()
            loss_cla_s.backward()
            for opt in self.optimizers.values():
                opt.update()
            n_batch += 1
            loss_cla_s_data += loss_cla_s.data
            acc_s_data += acc_s.data
            if n_batch >= total_batches:
                break
        loss_cla_s_data /= n_batch
        acc_s_data /= n_batch

        chainer.reporter.report({'acc_s': acc_s_data})
        chainer.reporter.report({'loss_cla_s': loss_cla_s_data})
        if not self.source_only:
            chainer.reporter.report({'loss_rec': loss_rec_data})
