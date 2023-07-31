#!/bin/bash

for cfg in swin_r2i1p1f1_p4_e768_wr32_lr1em3 swin_r2i1p1f1_p4_e768_wr64_lr1em3
do
   name=${cfg}
   for i in {1..3}; do sbatch --job-name=$name --dependency=singleton submit_batch.sh $cfg; done
   echo Submitted $name
done


# 80GB A100
for cfg in swin_r2i1p1f1_p2_e192_wr64_lr1em3 swin_r2i1p1f1_p4_e1024_wr64_lr1em3
do
   name=${cfg}
   for i in {1..5}; do sbatch --job-name=$name --dependency=singleton -C gpu\&hbm80g submit_batch.sh $cfg; done
   echo Submitted $name
done

