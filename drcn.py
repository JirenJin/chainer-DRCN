"""DRCN main class."""

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda


def clip_relu(x):
    xp = cuda.get_array_module(x)
    h = F.relu(x)
    return F.minimum(h, xp.ones_like(h.data))

class DRCN(chainer.Chain):
    def __init__(self, num_class=10,
                 kernel_size=(3, 3), pool_size=(2, 2), dropout=0.5, bn=True,
                 output_activation='softmax'):
        super().__init__()
        with self.init_scope():
            # encoder
            self.conv1 = L.Convolution2D(1, 100, ksize=(3, 3), pad=1)
            self.conv2 = L.Convolution2D(100, 150, ksize=(3, 3), pad=1)
            self.conv3 = L.Convolution2D(150, 200, ksize=(3, 3), pad=1)
            self.fc1 = L.Linear(200*8*8, 1000)
            # decoder
            self.fc2 = L.Linear(1000, 1000)
            self.fc3 = L.Linear(1000, 200*8*8)
            self.conv4 = L.Convolution2D(200, 200, ksize=(3, 3), pad=1)
            self.conv5 = L.Convolution2D(200, 150, ksize=(3, 3), pad=1)
            self.conv6 = L.Convolution2D(150, 100, ksize=(3, 3), pad=1)
            self.conv7 = L.Convolution2D(100, 1, ksize=(3, 3), pad=1)

    def encode(self, x):
        h = x
        h = F.max_pooling_2d(F.relu(self.conv1(h)), (2, 2))
        h = F.max_pooling_2d(F.relu(self.conv2(h)), (2, 2))
        h = F.relu(self.conv3(h))
        h = F.relu(self.fc1(h))
        return h

    def decode(self, x):
        h = x
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        h = F.reshape(h, [h.shape[0], 200, 8, 8])
        h = F.relu(self.conv4(h))
        # use cover_all=False, otherwise the output dimension will be smaller
        h = F.unpooling_2d(F.relu(self.conv5(h)), (2, 2), cover_all=False)
        h = F.unpooling_2d(F.relu(self.conv6(h)), (2, 2), cover_all=False)
        # use clip_relu, because the output should be images (0~1)
        h = clip_relu(self.conv7(h))
        return h
