from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyTrainDataset,MyValDataset,MyTestDataset,MyNIH,MyInference
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


# Configs
# resume_path = './models/control_sd21_ini.ckpt'
# resume_path = '/mnt/2TBHDD/leo/ControlNet/LLAMA_result/checkpoint/20epoch_sd_unlocked.ckpt'
resume_path = '/mnt/2TBHDD/leo/ControlNet/LLAMA_result/checkpoint/20epoch_prompt_sd_unlocked.ckpt'
# resume_path = '/home/leo/model/reversebonesuppression200epoch.ckpt'
batch_size = 1
logger_freq = 100000  # 500
learning_rate = 1e-5
sd_locked = False
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v21.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = MyInference()
# dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
train_dataloader = DataLoader(dataset, num_workers=10, batch_size=batch_size, shuffle=True)
# dataset = MyValDataset()
# val_dataloader = DataLoader(dataset, num_workers=10, batch_size=batch_size, shuffle=True)

# dataset = MyNIH()
# dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)

logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=[0], max_epochs=1, precision=32, callbacks=[logger])


# Train!
trainer.fit(model, train_dataloader)
# trainer.fit(model, train_dataloader)
# trainer.save_checkpoint("/mnt/2TBHDD/leo/ControlNet/LLAMA_result/checkpoint/20epoch_prompt_sd_unlocked.ckpt")