import os


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
