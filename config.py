import argparse

parser = argparse.ArgumentParser()


def get_config():
    config, unparsed = parser.parse_known_args()
    return config


def str2bool(v):
    if v.lower() in ('yes', 'true', 'y', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'n', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser.add_argument('--kernel_size', type=int, default=3, help='Kernel size of convolution layers in VGG')
parser.add_argument('--stride_size', type=int, default=2, help='stride size of convolution layers in VGG')
parser.add_argument('--dropout_rate', type=float, default=0.5, help='Rate of dropout in VGG')


parser.add_argument('--batch_size', type=float, default=10, help='Batch size')
parser.add_argument('--learning_rate', type=float, default=10e-5, help='Learning rate for training')
parser.add_argument('--epoch', type=int, default=10, help='The number of epoch')