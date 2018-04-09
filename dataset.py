import chainer
from chainer.datasets import TransformDataset
from chainercv.transforms import resize


def load_mnist():
    """Load MNSIT handwritten digit images in 32x32.

    Return:
        train dataset, test dataset
        in [n, c, h, w] format.
    """
    train, test = chainer.datasets.get_mnist(ndim=3, rgb_format=False)
    def transform(data):
        img, label = data
        img = resize(img, [32, 32])
        return img, label
    train = TransformDataset(train, transform)
    test = TransformDataset(test, transform)
    return train, test


def load_svhn():
    """Load grayscaled SVHN digit images.

    Return:
        train dataset, test dataset
        in [n, c, h, w] format.
    """
    train, test = chainer.datasets.get_svhn()
    def transform(data):
        img, label = data
        img = img[0] * 0.2989 + img[1] * 0.5870 + img[2] * 0.1140
        img = img.reshape(1, 32, 32)
        return img, label
    train = TransformDataset(train, transform)
    test = TransformDataset(test, transform)
    return train, test
