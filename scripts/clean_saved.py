import shutil
import os
import re
from os.path import basename, dirname, abspath
from glob import glob

import argparse


def num_checkpoints(path):
    files = glob(os.path.join(path, '*.pth'))
    return len(files)


def collect_satisfied(args):
    collected = []
    arch_paths = sorted(glob(os.path.join(args.saved_dir, args.pattern)))
    for arch_path in arch_paths:
        assert os.path.isdir(arch_path)
        if os.path.basename(arch_path) == 'runs':
            continue
        exp_paths = sorted(glob(os.path.join(arch_path, '*')))
        assert all([os.path.isdir(exp_path) for exp_path in exp_paths])

        exp_paths = [
            exp_path for exp_path in exp_paths
            if num_checkpoints(exp_path) < args.at_least
        ]
        collected.extend(exp_paths)
    return collected


def ask_one_by_one(args, collected):
    for path in collected:
        module_name = basename(dirname(path))
        exp_time = basename(path)

        runs_dir = os.path.join(args.saved_dir, 'runs', module_name, exp_time)
        print('\nDelete the following directories?')
        print(path)
        if os.path.exists(runs_dir):
            print(runs_dir)
        response = input('[y/N]? ')
        if response == 'y':
            shutil.rmtree(path)
            if os.path.exists(runs_dir):
                shutil.rmtree(runs_dir)
            print('Deleted.')
        else:
            print('No deletion performed.')


def ask_all_in_once(args, collected):
    to_delete = []
    print('The following directories will be deleted.')
    for path in collected:
        module_name = basename(dirname(path))
        exp_time = basename(path)
        to_delete.append(path)
        print(path)

        runs_dir = os.path.join(args.saved_dir, 'runs', module_name, exp_time)
        if os.path.exists(runs_dir):
            to_delete.append(runs_dir)
            print(runs_dir)
        print()

    response = input('[y/N]? ')
    if response == 'y':
        for path in to_delete:
            shutil.rmtree(path)
        print('Deleted')
    else:
        print('No deletion performed.')


parser = argparse.ArgumentParser()
parser.add_argument(
    '--at_least', type=int, default=1,
    help='The number of saved checkpoints required for not being deleted.'
         'In other words, the saved files whose #checkpoints are less than this number will be deleted (Default: 1)')
parser.add_argument(
    '-p', '--pattern', type=str,
    help='Those saved files fit this re pattern will be deleted.')
parser.add_argument(
    '--saved_dir', type=str,
    default=os.path.join(dirname(dirname(abspath(__file__))), 'saved'),
    help='The save directory'
)
parser.add_argument(
    '-all', '--all_in_once', action='store_true',
    help='Set this to disable the one-by-one comformation'
)
args = parser.parse_args()
assert args.pattern is not None, 'Pattern must be provided.'


if __name__ == '__main__':
    collected = collect_satisfied(args)
    if args.all_in_once:
        ask_all_in_once(args, collected)
    else:
        ask_one_by_one(args, collected)

