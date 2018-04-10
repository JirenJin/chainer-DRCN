# chainer-DRCN
Chainer Implementation of the paper "Deep Reconstruction-Classification Networks for Unsupervised Domain Adaptation" (DRCN).

## Usage
```
python train.py
```

## Optional arguments:
```
  -h, --help            show this help message and exit
  --device  # GPU to use
  --max_iter  # maximum iteration for training (epoch)
  --interval  # interval to evaluate/snapshot the model
  --batchsize  # mini-batch size
  --lr learning  # rate
  --out  # directory for experiment outputs
  --unit {iteration,epoch}
  --noise {no_noise,impulse,gaussian}  # whether to add noise to target domain images
  --source_only  # if this is True (1), train the model without using target domain data (without adaptation)
```
