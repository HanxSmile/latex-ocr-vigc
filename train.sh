nohup srun -p s2_bigdata --gres=gpu:1 --quotatype=reserved  \
torchrun --nproc-per-node 1 --master_port 29523  train.py  \
--cfg-path  /mnt/lustre/hanxiao/work/vigc/vigc/projects/train_latex_ocr.yaml  \
>> /mnt/lustre/hanxiao/input/latex_ocr_debug.log  2>&1 &