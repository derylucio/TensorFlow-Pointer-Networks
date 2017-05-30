import os
import numpy as np

with open('runbatch', 'r') as myfile:
    template = myfile.read()

partition = "40"
load_from_ckpts = "False" # "False"
cell_type = "GRU"
one_hot = "True"
lrs = [1e-4] #np.random.uniform(5e-4, 5e-6, 3)
rnns  = [400]#[100, 300, 400] 
fc_dims = [256]
regs = [0.0]#, 1e-2, 1e-3]
fnumber = 0
for lr in lrs:
    for rnn in rnns:
       for fc_dim in fc_dims:
           for reg in regs:
               attach = "srun --partition=k" + partition + " --gres=gpu:2 python main.py "
               attach += "--learning_rate " + str(lr) + " --rnn_size " + str(rnn) + " --fc_dim " + str(fc_dim) + " --reg " \
                          + str(reg) + " --load_from_ckpts " + load_from_ckpts + " --cell_type " + cell_type + " --encoder_attn_1hot " + one_hot
               newfile = template + "\n" + attach	
               with open("runconfig" + str(fnumber), "w+") as nfile:
                   nfile.write(newfile)
               cmd = "sbatch runconfig" + str(fnumber)
               os.system(cmd)
               fnumber += 1
             
