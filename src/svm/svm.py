"""Command line entry point for the rice shape and SVM pipeline."""

import argparse
import importlib
import sys


def main():
    parser = argparse.ArgumentParser(description='Rice classification with shape features and SVM.')
    parser.add_argument('command', choices=['train', 'evaluate', 'predict', 'download'])
    argv = sys.argv[1:]
    args = parser.parse_args(argv[:1])
    remaining = argv[1:]
    module = 'download_dataset' if args.command == 'download' else args.command
    importlib.import_module(f'{__package__}.{module}').main(remaining)


if __name__ == '__main__':
    main()
