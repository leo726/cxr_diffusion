from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyInference
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import numpy as np
import torch
import torchvision
from PIL import Image

def gen_cxr_path(prompt):
    resume_path = '/mnt/sda/yangling/mimic_controlnet/ckpt/20+20epoch_prompt_sd_unlocked_10epochfinetune.ckpt'
    batch_size = 1
    logger_freq = 1
    learning_rate = 1e-5
    sd_locked = False
    only_mid_control = False


    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('./models/cldm_v21.yaml').cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control
    model.to(torch.device("cuda:0"))

    # create dataset
    x = torch.rand(1,256,256,3)
    batch = {'jpg':x,'txt':[prompt],'hint':x}
    log = model.log_images(batch)
    grid = torchvision.utils.make_grid(log['samples_cfg_scale_1.10'], nrow=4)
    grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
    grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
    grid = grid.cpu().numpy()
    grid = (grid * 255).astype(np.uint8)
    path = './gen_cxr.png'
    Image.fromarray(grid).save(path)
    return path

img_path = gen_cxr_path('Lateral view of the chest was obtained. The previously seen multifocal bibasilar airspace opacities have almost completely resolved with only slight scarring seen at the bases. There are new ill-defined bilateral linear opacities seen in the upper lobes, which given their slight retractile behavior are likely related to radiation fibrosis.')
print(img_path)