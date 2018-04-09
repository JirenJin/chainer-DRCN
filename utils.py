"""
Contains all helpers for DRCN
"""

def transform_factory(size=None, pixel_mean=None, grayscale=False, scale=1.0,
                      crop_size=None, mirror=False, is_train=False):
    def transform(in_data):
        img, label = in_data
        if grayscale:
            # in case the img is already grayscale
            if img.shape[0] != 1:
                img = (img[0] * 0.2989 + img[1] * 0.5870 +
                       img[2] * 0.1140).reshape(1, img.shape[1], img.shape[2])
        if pixel_mean is not None:
            img -= pixel_mean
        if size is not None:
            # avoid unnecessary computation
            if (size, size) != img.shape[1:]:
                img = resize(img, (size, size))
        if crop_size is not None:
            if is_train:
                img = random_crop(img, (crop_size, crop_size))
            else:
                img = center_crop(img, (crop_size, crop_size))
        if mirror and is_train:
            img = random_flip(img, x_random=True)
        img *= scale
        return img, label
    return transform


def parse_args(args):
    if args.config_file:
        data = yaml.load(args.config_file)
        delattr(args, 'config_file')
        for key, value in data.items():
            if hasattr(args, key):
                setattr(args, key, value)
            else:
                print("{} is ignored, please check the key name.".format(key))
    return args
