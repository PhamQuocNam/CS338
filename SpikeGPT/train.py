########################################################################################################
# The RWKV v2-RNN Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import glob
import logging
import os
import threading
import time
import datetime
import json
from src.model import GPT, GPTConfig
from src.trainer import Trainer, TrainerConfig
from src.utils import Dataset
import torch
import numpy as np
from src.spikingjelly.clock_driven import functional
from src.binidx import MMapIndexedDataset
from accelerate import accelerator
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


### Step 1: set training data ##########################################################################

datafile_train = "/kaggle/input/datasets/nviethoang/50kfixdataagencyspikegpt/train_tool_data_copyfix_text_document"
datafile_valid = "/kaggle/input/datasets/nviethoang/50kfixdataagencyspikegpt/valid_tool_data_copyfix_text_document"
datafile_test = "test.txt"
datafile_encoding = 'utf-8'
# datafile_encoding = 'utf-16le'

### Step 2: set model size #############################################################################

ctx_len = 1024        # ===> increase T_MAX in model.py if your ctx_len > 1024
n_layer = 18
n_embd = 768

# 'RWKV' (better for char-level English) or 'RWKV-ffnPre' (better in some cases)
model_type = 'RWKV'

### Step 3: set batch size #############################################################################

# ===> batch_size must be divisible by B_GROUP_FORWARD and B_GROUP_BACKWARD in model.py
# For example, if your batch_size = 20, you can set B_GROUP_FORWARD = 4, B_GROUP_BACKWARD = 2
# If you see "CUDA out of memory", reduce it. Use GPU-Z to find the highest value for your VRAM.
batch_size = 2

### Step 4: set learning rate, training mini-epochs #######################################################

lr_init = 1e-5
lr_final = 1e-6
# the mini-epoch is very short and of fixed length (ctx_len * epoch_length_fixed tokens)
n_epoch = 200
# 0 = never, 1 = every mini-epoch, 2 = every two mini-epochs, etc.
epoch_save_frequency = 10
epoch_save_path = 'updated_2_model_weights/'

epoch_length_fixed = 1000

########################################################################################################

import src.utils
src.utils.set_seed(42) # remember to change seed if you load a model

np.set_printoptions(precision=4, suppress=True, linewidth=200)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO,)

grad_norm_clip = 1.0
warmup_tokens = 0

betas = (0.9, 0.99)
eps = 4e-9

num_workers = 2

########################################################################################################
# Load data
########################################################################################################

print('loading data... ' + datafile_train)
# train_dataset = Dataset(open(
#     datafile_train, "r", encoding=datafile_encoding).read(), ctx_len, epoch_length_fixed)

train_dataset = Dataset(MMapIndexedDataset(datafile_train), ctx_len, epoch_length_fixed) #use it when you use binidx files
valid_dataset = Dataset(MMapIndexedDataset(datafile_valid), ctx_len, epoch_length_fixed)
# valid_dataset = Dataset(open(
#     datafile_valid, "r", encoding=datafile_encoding).read(), ctx_len, epoch_length_fixed) 

# test_dataset = Dataset(open(
#     datafile_test, "r", encoding=datafile_encoding).read(), ctx_len, epoch_length_fixed)
########################################################################################################
# Train model
########################################################################################################
if __name__ == '__main__':
    model = GPT(GPTConfig(train_dataset.vocab_size, train_dataset.ctx_len, model_type=model_type,
                          n_layer=n_layer, n_embd=n_embd)).cuda()

    pretrained = os.environ.get(
        "SPIKEGPT_CHECKPOINT",
        "/kaggle/input/models/hykhangg/spikegpt216m/pytorch/default/1/SpikeGPT-216M.pth",
    )
    if os.path.isfile(pretrained):
        print("\n" + "="*50)
        print(f"ĐÃ TÌM THẤY FILE PRE-TRAINED TẠI: {pretrained}")
        m2 = torch.load(pretrained, map_location=torch.device("cpu"))
        load_result = model.load_state_dict(m2, strict=False)
        print(f"NẠP THÀNH CÔNG! Các layer chưa có tạ (sẽ khởi tạo random):")
        print(load_result.missing_keys) # Sẽ in ra head_q và head_k
        print("="*50 + "\n")
    else:
        print("\n" + "!"*50)
        print(f"CẢNH BÁO: KHÔNG TÌM THẤY FILE {pretrained} !!!")
        print("QUÁ TRÌNH TRAIN SẼ BẮT ĐẦU VỚI RANDOM WEIGHTS (FROM SCRATCH)!!!")
        print("!"*50 + "\n")


    test_dataset = None
    print('model', model_type, 'epoch', n_epoch, 'batchsz', batch_size, 'betas',
          betas, 'eps', eps, 'ctx', ctx_len, 'layer', n_layer, 'embd', n_embd, )
    tconf = TrainerConfig(model_type=model_type, max_epochs=n_epoch, batch_size=batch_size,
                          learning_rate=lr_init, lr_decay=True, lr_final=lr_final, betas=betas, eps=eps, grad_norm_clip=grad_norm_clip,
                          warmup_tokens=warmup_tokens, final_tokens=n_epoch*len(train_dataset)*ctx_len, num_workers=num_workers, epoch_save_frequency=epoch_save_frequency, epoch_save_path=epoch_save_path)
    trainer = Trainer(model, train_dataset, valid_dataset, test_dataset, tconf)
    os.makedirs(epoch_save_path, exist_ok=True)

    def cleanup_checkpoints(save_dir, keep=5):
        while True:
            time.sleep(60)
            if not os.path.exists(save_dir):
                continue
            files = glob.glob(os.path.join(save_dir, "*.pth"))
            if len(files) > keep:
                files.sort(key=os.path.getmtime)
                for ckpt in files[:-keep]:
                    try:
                        os.remove(ckpt)
                    except OSError:
                        pass

    threading.Thread(target=cleanup_checkpoints, args=(epoch_save_path, 5), daemon=True).start()
    trainer.train()

    torch.save(model.state_dict(), 'trained-' + str(n_epoch) + '-' + trainer.get_run_name() +
               '-' + datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S') + '.pth')