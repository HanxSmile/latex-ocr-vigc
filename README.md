## Installation

1. (Optional) Creating conda environment

```bash
conda create -n vigc python=3.8
conda activate vigc
```

3. For development, you should build from source

```bash
git clone https://gitlab.pjlab.org.cn/fdc/mllm/vigc.git -b latex-ocr-debug
cd vigc
pip install -e .
```

## Train & Eval Latex-OCR
```bash
srun -p s2_bigdata --gres=gpu:4 --quotatype=reserved  torchrun --nproc-per-node 4 --master_port 29523  train.py  --cfg-path vigc/projects/train_latex_ocr.yaml
```



## Only Eval Latex-OCR

```bash
srun -p s2_bigdata --gres=gpu:4 --quotatype=reserved  torchrun --nproc-per-node 4 --master_port 29523  evaluate.py  --cfg-path vigc/projects/eval_latex_ocr.yaml
```



## Data

| best model | `/mnt/petrelfs/share_data/hanxiao/latex-ocr/checkpoint_best.pth` |
| ---------- | ------------------------------------------------------------ |
| dataset    | `/mnt/petrelfs/share_data/hanxiao/latex-ocr/pdf`             |



## Performance

| BLEU score         | normed edit distance |
| ------------------ | -------------------- |
| 0.9163332373184626 | 0.06568927457913339  |

