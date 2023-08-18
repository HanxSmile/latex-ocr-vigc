# VIGC: Visual Instruction Generation and Correction

We propose **Visual Instruction Generation and Correction (VIGC)**, a framework capable of autonomously generating high-quality image-text instruction fine-tuning datasets.

<p align="center">
    <br>
    <img src="assets/overview.png"/>
    <br>
<p>

## Table of Contents
  - [Getting Started](#getting-started)
    - [Installation](#installation)
    - [Prepare Models](#prepare-models)
    - [Launching Demo](#launching-demo)
  - [Tutorials](#tutorials)
    - [Generate QA](#generate-qa)
    - [Train VIGC Model](#train-vigc-model)
    - [Train VQA Model](#train-vqa-model)
  - [Documentations](#documentations)
  - [Paper and Citing VIGC](#paper-and-citing-vigc)
  - [License](#license)

## Getting Started

### Installation

1. (Optional) Creating conda environment

   ```bash
   conda create -n vigc python=3.8
   conda activate vigc
   ```

2. Install mmpretrain

   you can follow the [tutorial](https://github.com/open-mmlab/mmpretrain#installation)

3. You may build from source

   ```bash
   git clone https://gitlab.pjlab.org.cn/fdc/mllm/vigc.git
   cd vigc
   pip install -e .
   ```

### Prepare Models
1. obtain vicuna model

   Vicuna is an open-source LLAMA-based LLM that has a performance close to ChatGPT. We currently use the v1.1 version of Vicuna-13B and 7B. If you already have the Vicuna weights with correct version, modify the `llm_model` in [Model Config](vigc/configs/models/mini_gpt4_vicuna13b.yaml) to the folder that contains your Vicuna weights. Otherwise, you can follow [this instruction](GetVicuna.md) to get them, remenber that modify the config file too.

2. download pretrain model

   We support two different kinds of pretrain checkpoints to load from: minigpt-4 and instrucblip. You can download them from the table below, more details please visit their original repositories: [minigpt-4](https://github.com/Vision-CAIR/MiniGPT-4/blob/main/README.md##3) and [instrucblip](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip).

   | Model Type | Checkpoint pretrained with Vicuna 7B | Checkpoint pretrained with Vicuna 13B |
   :-------------------:|:--------------------:|:------------------:
   minigpt-4 | [Download]() | [Download]() 
   instrucblip | [Download]() | / 

   After download the pretrained checkpoints, please modify the `finetuned` in [Model Config](vigc/configs/models/mini_gpt4_vicuna13b.yaml) to the folder that contains pretrian weights.

3. download fintuned vigc model

   Download the pretrained vigc checkpoints according to fintuned dataset and the Vicuna model you prepared.
   | Fintuned Dataset | Checkpoint Fintuned with Vicuna 7B | Checkpoint Fintuned with Vicuna 13B |
   :-------------------:|:--------------------:|:------------------:
   LLaVA | [Download]() | [Download]() 
   OKVQA | [Download]() | [Download]() 
   A-OKVQA | [Download]() | [Download]() 


### Launching Demo

   To Launch a demo locally, run ```bash vigc_demo.sh``` and then follow the instruction on the prompts to view in browser. Arguments are as follows:
   - device0: The gpu id of the first model
   - device1: The gpu id of the second model
   - ckpt_minigpt4: The checkpoint file of the Mini-GPT4 model
   - ckpt_instruct_blip: The checkpoint file of the Instruct Blip model

## Tutorials


### Generate QA

1. generate QA based on COCO2017 for Llava

   1. You should first download the [finetuned checkpoint file](#ready-made-vigc-model) (Mini-GPT4 vicuna7b or vicuna13b)
   2. Then modify the `finetuned` in corresponding Inference Config to the path to the checkpoint file.

   ```bash
   torchrun --nproc_per_node=8 evaluate.py --cfg-path vigc/projects/mini_gpt4_vicuna7b/generate_qa/llava-150k/generate_llava_qa_conv.yaml   # generate conversation data for Llava using MiniGPT4-vicuna7b
   
   torchrun --nproc_per_node=8 evaluate.py --cfg-path vigc/projects/mini_gpt4_vicuna7b/generate_qa/llava-150k/generate_llava_qa_detail.yaml   # generate detail description data for Llava using MiniGPT4-vicuna7b
   
   torchrun --nproc_per_node=8 evaluate.py --cfg-path vigc/projects/mini_gpt4_vicuna7b/generate_qa/llava-150k/generate_llava_qa_complex.yaml   # generate complex reasoning data for Llava using MiniGPT4-vicuna7b
   
   torchrun --nproc_per_node=8 evaluate.py --cfg-path vigc/projects/mini_gpt4_vicuna13b/generate_qa/llava-150k/generate_llava_qa_conv.yaml   # generate conversation data for Llava using MiniGPT4-vicuna13b
   
   torchrun --nproc_per_node=8 evaluate.py --cfg-path vigc/projects/mini_gpt4_vicuna13b/generate_qa/llava-150k/generate_llava_qa_detail.yaml   # generate detail description data for Llava using MiniGPT4-vicuna13b
   
   torchrun --nproc_per_node=8 evaluate.py --cfg-path vigc/projects/mini_gpt4_vicuna13b/generate_qa/llava-150k/generate_llava_qa_complex.yaml   # generate complex reasoning data for Llava using MiniGPT4-vicuna13b
   ```

2. generate QA based on Object365 for Llava

   1. You should first download the [finetuned checkpoint file](#ready-made-vigc-model) (Mini-GPT4 vicuna7b or vicuna13b)
   2. Then modify the `finetuned` in corresponding Inference Config to the path to the checkpoint file.

   ```bash
   torchrun --nproc_per_node=8 evaluate.py --cfg-path vigc/projects/mini_gpt4_vicuna7b/generate_qa/llava-150k/generate_llava_qa_object365_conv.yaml   # generate conversation data for Llava using MiniGPT4-vicuna7b
   
   torchrun --nproc_per_node=8 evaluate.py --cfg-path vigc/projects/mini_gpt4_vicuna7b/generate_qa/llava-150k/generate_llava_qa_object365_detail.yaml  # generate detail description data for Llava using MiniGPT4-vicuna7b
   
   torchrun --nproc_per_node=8 evaluate.py --cfg-path vigc/projects/mini_gpt4_vicuna7b/generate_qa/llava-150k/generate_llava_qa_object365_complex.yaml   # generate complex reasoning data for Llava using MiniGPT4-vicuna7b
   
   torchrun --nproc_per_node=8 evaluate.py --cfg-path vigc/projects/mini_gpt4_vicuna13b/generate_qa/llava-150k/generate_llava_qa_object365_conv.yaml   # generate conversation data for Llava using MiniGPT4-vicuna13b
   
   torchrun --nproc_per_node=8 evaluate.py --cfg-path vigc/projects/mini_gpt4_vicuna13b/generate_qa/llava-150k/generate_llava_qa_object365_detail.yaml   # generate detail description data for Llava using MiniGPT4-vicuna13b
   
   torchrun --nproc_per_node=8 evaluate.py --cfg-path vigc/projects/mini_gpt4_vicuna13b/generate_qa/llava-150k/generate_llava_qa_object365_complex.yaml   # generate complex reasoning data for Llava using MiniGPT4-vicuna13b
   ```

3. generate QA based on COCO2017 for A-OKVQA or OKVQA

   1. You should first download the  [finetuned checkpoint file](#ready-made-vigc-model) (InstructBlip vicuna7b)

   2. Then modify the `pretrained` in corresponding Inference Config to the path to the checkpoint file.

   3. Generate the question first:

      ```bash
      torchrun --nproc_per_node=8 evaluate.py --cfg-path vigc/projects/instruct_blip_vicuna7b/generate_qa/a-okvqa/generate_question.yaml   # generate questions for A-OKVQA using instruct-blip-vicuna7b
      
      torchrun --nproc_per_node=8 evaluate.py --cfg-path vigc/projects/instruct_blip_vicuna7b/generate_qa/okvqa/generate_question.yaml   # generate questions for OKVQA using instruct-blip-vicuna7b
      ```

   4. Modify the `annotaion` in `generate_answer.yaml` to the path of the questions generated in the above step, then generate the answers: 

      ```bash
      torchrun --nproc_per_node=8 evaluate.py --cfg-path vigc/projects/instruct_blip_vicuna7b/generate_qa/a-okvqa/generate_answer.yaml   # generate answers for A-OKVQA using instruct-blip-vicuna7b
      
      torchrun --nproc_per_node=8 evaluate.py --cfg-path vigc/projects/instruct_blip_vicuna7b/generate_qa/okvqa/generate_answer.yaml   # generate answers for OKVQA using instruct-blip-vicuna7b
      ```

### Train VIGC Model

1. Finetune Instruct Blip to train a A-OKVQA VIGC model

   ```python
   torchrun --nproc_per_node=8 train.py --cfg-path vigc/projects/instruct_blip_vicuna7b/vig/a-okvqa/normal_vqga.yaml
   ```

2. Finetune Instruct Blip to train a OKVQA VIGC model

   ```bash
   torchrun --nproc_per_node=8 train.py --cfg-path vigc/projects/instruct_blip_vicuna7b/vig/okvqa/normal_vqga.yaml
   ```

3. Finetune Mini-GPT4 to train a Llava instruct 150k VIGC model

   ```bash
   torchrun --nproc_per_node=8 train.py  --cfg-path vigc/projects/mini_gpt4_vicuna7b/vig/llava-150k/normal_vqga.yaml  # using Mini-GPT4 Vicuna7b
   
   torchrun --nproc_per_node=8 train.py  --cfg-path vigc/projects/mini_gpt4_vicuna13b/vig/llava-150k/normal_vqga.yaml  # using Mini-GPT4 Vicuna13b
   ```

### Train VQA Model

1. Train a baseline model of A-OKVQA using Instruct Blip

   ```bash
   torchrun --nproc_per_node=8 train.py  --cfg-path vigc/projects/instruct_blip_vicuna7b/vqa/a-okvqa/normal_vqa.yaml  # using Instructblip Vicuna7b
   ```

2. Make use of VIGC data to train a better model of A-OKVQA using Instruct Blip

   ```bash
   torchrun --nproc_per_node=8 train.py  --cfg-path vigc/projects/instruct_blip_vicuna7b/vqa/a-okvqa/coco_pseudo_vqa.yaml  # using Instructblip Vicuna7b
   ```

3. Train a baseline model of OKVQA using Instruct Blip

   ```bash
   torchrun --nproc_per_node=8 train.py  --cfg-path vigc/projects/instruct_blip_vicuna7b/vqa/okvqa/normal_vqa.yaml  # using Instructblip Vicuna7b
   ```

4. Make use of VIGC data to train a better model of OKVQA using Instruct Blip

   ```bash
   torchrun --nproc_per_node=8 train.py  --cfg-path vigc/projects/instruct_blip_vicuna7b/vqa/okvqa/coco_pseudo_vqa.yaml  # using Instructblip Vicuna7b
   ```


## Documentations
For more details and advanced usages, please refer to
[documentation](https://opensource.salesforce.com/LAVIS//latest/index.html#).


## Paper and Citing VIGC
You can find more details in our [paper](https://arxiv.org/abs/2209.09019).

If you're using VIGC in your research or applications, please cite using this BibTeX:
```bibtex
@misc{li2022lavis,
      title={LAVIS: A Library for Language-Vision Intelligence}, 
      author={Dongxu Li and Junnan Li and Hung Le and Guangsen Wang and Silvio Savarese and Steven C. H. Hoi},
      year={2022},
      eprint={2209.09019},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Contact us
If you have any questions, comments or suggestions, please do not hesitate to contact us at lavis@salesforce.com.

## License
[BSD 3-Clause License](LICENSE.txt)
