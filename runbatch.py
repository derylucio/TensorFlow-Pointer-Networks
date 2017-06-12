import os
import numpy as np

partition = "80"
with open('runbatch', 'r') as myfile:
    template = myfile.read()

load_from_ckpts = "False" # "False"
cell_type = "GRU"
one_hot = "True"
lrs = [3e-5]    
rnns  = [1000]  
fc_dims = [256] 
regs = [0.001] 
fnumber = 0
dps = [-1]      
puzzle_height = 2
puzzle_width = 2
max_steps = puzzle_height * puzzle_width

for lr in lrs:
   for reg in regs:
       for fc_dim in fc_dims:
            for rnn in rnns:
               for dp in dps:
                   attach = "srun --partition=k" + partition + " --gres=gpu:2 python main.py "
                   attach += "--learning_rate " + str(lr) + " --rnn_size " + str(rnn) + " --fc_dim " + str(fc_dim) + " --reg " + str(reg) + " --dp " + str(dp)\
                           + " --load_from_ckpts " + load_from_ckpts + " --cell_type " + cell_type + " --encoder_attn_1hot " + one_hot\
                           + " --puzzle_height " + str(puzzle_height) + " --puzzle_width " + str(puzzle_width)\
                           + " --max_steps " + str(max_steps)
                   newfile = template + "\n" + attach	
                   with open("runconfig" + str(fnumber), "w+") as nfile:
                       nfile.write(newfile)
                   cmd = "sbatch runconfig" + str(fnumber)
                   os.system(cmd)
                   fnumber += 1
             

