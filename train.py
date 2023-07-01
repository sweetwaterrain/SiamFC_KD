from __future__ import absolute_import

import os
from got10k.datasets import *

from siamfc import TrackerSiamFC

if __name__ == '__main__':
    # root_dir = os.path.expanduser('~/data/GOT-10k')
    # seqs = GOT10k(root_dir, subset='train', return_meta=True)

    root_dir = os.path.expanduser('/kaggle/input/otb2015/OTB100')
    seqs = OTB(root_dir, version=2015)

    tracker = TrackerSiamFC()
    tracker.train_over(seqs)
    
