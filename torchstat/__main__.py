from torchstat import stat
import argparse
import importlib.util
import torch


def arg():
    parser = argparse.ArgumentParser(description='Torch model statistics')
    parser.add_argument('--file', '-f', type=str,
                        help='Module file.')
    parser.add_argument('--model', '-m', type=str,
                        help='Model name')
    parser.add_argument('--size', '-s', type=str, default='3x224x224',
                        help='Input size. channels x height x width (default: 3x224x224)')
    return parser.parse_args()


def main():
    args = arg()
    try:
        spec = importlib.util.spec_from_file_location('models', args.file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = getattr(module, args.model)()
    except Exception:
        import traceback
        print(f'Tried to import {args.model} from {args.file}. but failed.')
        traceback.print_exc()

        import sys
        sys.exit()

    input_size = tuple(int(x) for x in args.size.split('x'))
    stat(model, input_size, query_granularity=1)
