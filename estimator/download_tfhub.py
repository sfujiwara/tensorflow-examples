import argparse
import os
import tensorflow_hub as hub


parser = argparse.ArgumentParser()
parser.add_argument('--tfhub_dir', type=str)
args = parser.parse_args()

TFHUB_DIR = args.tfhub_dir


def main():
    os.environ['TFHUB_CACHE_DIR'] = TFHUB_DIR
    hub.Module('https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/1', trainable=False)


if __name__ == '__main__':
    main()
