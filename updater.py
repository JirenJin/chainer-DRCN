import chainer
import chainer.functions as F

from base_da_updater import BaseDAUpdater


class Updater(BaseDAUpdater):
    """DRCN Updater."""
    def __init__(self, s_iter, t_iter, optimizers, args):
        super().__init__(s_iter, t_iter, optimizers, device=args.device)
        self.model = optimizers['model'].target

    def update_core(self):
        # autoencoder training
        loss_rec_data = 0
        n_batch = 0
        total_batches = len(self.t_iter.dataset) / self.t_iter.batch_size
        for t_batch in self.t_iter:
            t_imgs, _ = self.converter(t_batch, self.device)
            t_encoding = self.model.encode(t_imgs)
            t_decoding = self.model.decode(t_encoding)
            loss_rec = F.mean_squared_error(t_decoding, t_imgs)
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
        chainer.reporter.report({'loss_rec': loss_rec_data})
