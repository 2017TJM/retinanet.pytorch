import os.path as osp
import sys
from pathlib import Path

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
        print('added {} to pythonpath'.format(path))

# Add pycocotools to PYTHONPATH
home_dir = str(Path.home())
coco_path = osp.join(home_dir, 'Github', 'v-coco')
add_path(coco_path)
