#!/bin/bash
#set a job name    
#SBATCH --job-name=jigsaws_pointer
#################  
#a file for job output, you can check job progress, append the job ID with %j to make it unique
#SBATCH --output=runlogs/jigsaws_pointer.%j.out
#################
# a file for errors from the job
#SBATCH --error=runlogs/jigsaws_pointer.%j.err
#################
#Quality of Service (QOS); think of it as job priority, there is also --qos=long for with a max job length of 7 days, qos normal is 48 hours.
# REMOVE "normal" and set to "long" if you want your job to run longer than 48 hours,  
# NOTE- in the hns partition the default max run time is 7 days , so you wont need to include qos
# We are submitting to the dev partition, there are several on sherlock: normal, gpu, bigmem (jobs requiring >64Gigs RAM) 
#SBATCH --time=4:30:00 --partition=k80 --gres=gpu:2

#################
#number of nodes you are requesting, the more you ask for the longer you wait
#SBATCH --nodes=1
 
source  .env/bin/activate
 
module load python/3.5.0
module load cudnn/5.1
module load cuda80/blas/8.0.44
module load cuda80/toolkit/8.0.44

