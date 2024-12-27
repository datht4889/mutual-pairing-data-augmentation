import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--ucl', action='store_true')
    args = parser.parse_args()
    return args

args = parse_arguments()
print(args.ucl)
