from __future__ import absolute_import

import os
from got10k.experiments import *

from siamfc import TrackerSiamFC


if __name__ == '__main__':
    net_path = '/kaggle/working/SiamFC_VanillaNet/pretrained/siamfc_alexnet_e10.pth'
    tracker = TrackerSiamFC(net_path=net_path)

    root_dir = os.path.expanduser('/kaggle/input/otb2015/OTB100')
    e = ExperimentOTB(root_dir, version=2015)
    e.run(tracker)
    e.report([tracker.name])
