import shutil
import os
from glob import glob


for direc in glob('saved/*/*'):
    splits = direc.split('/')
    if splits[1] == 'runs':
        continue
    hey = glob(os.path.join(direc, '*.pth'))
    if len(hey) == 0:
        splits.insert(1, 'runs')
        runs_dir = '/'.join(splits)
        print(direc)
        print(runs_dir)
        try:
            shutil.rmtree(direc)
            shutil.rmtree(runs_dir)
        except Exception as e:
            print(str(e))
