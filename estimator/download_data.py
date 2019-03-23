import argparse
import tensorflow_datasets as tfds


parser = argparse.ArgumentParser()
parser.add_argument('--tfds_dir', type=str)
args = parser.parse_args()

TFDS_DIR = args.tfds_dir


def main():
    builder = tfds.builder('mnist', data_dir=TFDS_DIR)
    builder.download_and_prepare()
    return


if __name__ == '__main__':
    main()
