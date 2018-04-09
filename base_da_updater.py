import six

from chainer.dataset import convert
from chainer.training import _updater


class BaseDAUpdater(_updater.Updater):

    """Implementation of Updater specially for Domain Adaptation.

    Args:
        s_iter: Source domain iterator for the training dataset.
        t_iter: Target domain iterator for the training dataset.
        optimizers: Optimizers to update parameters.
        converter: Converter function to build input arrays. Each batch
            extracted by the iterators and the ``device`` option are passed
            to this function. :func:`~chainer.dataset.concat_examples` is used
            by default.
        device: Device to which the training data is sent. Negative value
            indicates the host memory (CPU).
        loss_scale (float): Loss scaling factor. Loss scaling is a usefull
            technique to mitigate vanishing gradient issue that tends to happen
            when low precision data type like float16 is used during training.
            If you set loss scaling factor, gradients of loss values are to be
            multiplied by the factor before backprop starts. The factor is
            propagated to whole gradients in a computational graph along the
            backprop. The gradients of parameters are divided by the factor
            just before the parameters are to be updated.

    Attributes:
        s_iter: source domain iterator for the training dataset.
        t_iter: target domain iterator for the training dataset.
        optimizers: Optimizers to update parameters.
        converter: Converter function.
        device: Device to which the training data is sent.
        iteration: Current number of completed updates.

    """

    def __init__(self, s_iter, t_iter, optimizers,
                 converter=convert.concat_examples, device=None,
                 loss_scale=None):
        self.s_iter = s_iter
        self.t_iter = t_iter
        self.optimizers = optimizers

        if device is not None and device >= 0:
            for optimizer in six.itervalues(self.optimizers):
                optimizer.target.to_gpu(device)

        self.converter = converter
        self.device = device
        self.iteration = 0

        self.loss_scale = loss_scale
        if loss_scale is not None:
            for optimizer in six.itervalues(self.optimizers):
                optimizer.set_loss_scale(loss_scale)

    @property
    def epoch(self):
        return self.s_iter.epoch

    @property
    def epoch_detail(self):
        return self.s_iter.epoch_detail

    @property
    def previous_epoch_detail(self):
        return self.s_iter.previous_epoch_detail

    @property
    def is_new_epoch(self):
        return self.s_iter.is_new_epoch

    def get_all_optimizers(self):
        return self.optimizers

    def finalize(self):
        """Finalizes the updater object.

        This method calls the `finalize` method of each iterator that
        this updater has.
        It is called at the end of training loops.

        """
        self.s_iter.finalize()
        self.t_iter.finalize()

    def update(self):
        """Updates the parameters of the models.

        This method implements an update formula for the training task,
        including data loading, forward/backward computations, and actual
        updates of parameters.
        This method is called once at each iteration of the training loop.

        """
        self.update_core()
        self.iteration += 1

    def update_core(self):
        raise NotImplementedError

    def serialize(self, serializer):
        """Serializes the current state of the updater object."""
        self.s_iter.serialize(serializer['iterator:source'])
        self.t_iter.serialize(serializer['iterator:target'])

        for name, optimizer in six.iteritems(self.optimizers):
            optimizer.serialize(serializer['optimizer:' + name])
            optimizer.target.serialize(serializer['model:' + name])

        self.iteration = serializer('iteration', self.iteration)
