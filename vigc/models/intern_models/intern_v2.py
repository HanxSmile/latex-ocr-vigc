import logging
import random
import copy
import math

from functools import partial
import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from vigc.common.registry import registry
from vigc.models.intern_models.blip2 import Blip2Base, disabled_train
from vigc.models.intern_models.hf_convert_llama import LlamaTokenizer as PyTokenizer
from vigc.models.intern_models.modeling_llama import LlamaForCausalLM
from vigc.models.intern_models.lora import LoRALinear
from transformers import LlamaTokenizer
from transformers.models.llama.configuration_llama import LlamaConfig
import transformers

transformers.logging.set_verbosity_error()

from transformers import StoppingCriteriaList
from .intern import StoppingCriteriaSub

meta_instruction = """meta instruction
You are an AI assistant whose name is 浦语.
- 浦语 is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- 浦语 can understand and communicate fluently in the language chosen by the user such as English and 中文.
conversation
"""


def merge_lora_func(model, model_ckpt, scale, path):
    print(f"Merge lora weight from {path}")
    for name, param in model.llama_model.named_parameters():
        if 'self_attn' in name and 'weight' in name:
            lora_a = 'llama_model.' + name.replace('.weight', '.lora_A.weight')
            lora_b = 'llama_model.' + name.replace('.weight', '.lora_B.weight')
            linear = 'llama_model.' + name
            lora_a_weight = model_ckpt.get(lora_a, None)
            lora_b_weight = model_ckpt.get(lora_b, None)
            if lora_a_weight is not None and lora_b_weight is not None:
                lora_a_weight = lora_a_weight.float()
                lora_b_weight = lora_b_weight.float()
                lora_weight = lora_b_weight @ lora_a_weight
                ori_weight = model_ckpt.get(linear).float()
                new_weight = ori_weight + scale * lora_weight.to(ori_weight.device)
                _ = model_ckpt.pop(lora_a)
                _ = model_ckpt.pop(lora_b)
                model_ckpt[linear] = new_weight
                print(f"merge lora of {name}")

                x = torch.rand(2, 1280, new_weight.shape[1]).cuda()
                LN = nn.LayerNorm(new_weight.shape[1], elementwise_affine=False).cuda()
                x = LN(x)
                l1 = nn.Linear(new_weight.shape[1], new_weight.shape[0], bias=False).cuda()
                l1.weight = nn.Parameter(new_weight.float().cuda())
                l1.eval()

                l2 = LoRALinear(new_weight.shape[1], new_weight.shape[0], bias=False).cuda()
                l2.lora_scaling = scale

                l2.weight = nn.Parameter(ori_weight.float().cuda())
                l2.lora_A.weight = nn.Parameter(lora_a_weight.float().cuda())
                l2.lora_B.weight = nn.Parameter(lora_b_weight.float().cuda())
                l2.eval()
                o1 = l1(x)
                o2 = l2(x)
                assert torch.allclose(o1, o2, atol=1e-3)
                print(f"merge check pass")
    msg = model.load_state_dict(model_ckpt, strict=False)
    print(msg)
    for name, param in model.llama_model.named_parameters():
        if 'lora_B' in name:
            # print (name, torch.mean(param))
            assert torch.mean(param) == 0  ### check the lora is init mode
    return model


@registry.register_model("intern_v2")
class Intern_v2(Blip2Base):
    """
    BLIP2 GPT-LLAMA model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/minigpt4.yaml",
    }

    def __init__(
            self,
            vit_model="eva_clip_g",
            q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
            img_size=224,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision="fp16",
            freeze_vit=True,
            freeze_qformer=True,
            freeze_qformer_query=True,
            use_qformer=True,
            freeze_llama_proj=False,
            instruct_blip=False,
            img_embed_unfold=1,
            num_query_token=32,
            llama_model="",
            prompt_path="",
            prompt_template="",
            max_txt_len=32,
            low_resource=False,  # use 8 bit and put vit in cpu
            end_sym='\n',
            load_in_8bit=True,
            device_map='auto',
            fully_open_llm=False,
            llama_lora=None,
            gradient_checkpointing=False,
            use_7132k=False,
            simple_conv_prompt=False,
            mask_human=True,
            meta_for_long=False,
            no_meta=False,
            balance_sample=False,
            only_instruct_ft=False,
            quant_pretrain=False,
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource
        self.meta_for_long = meta_for_long
        if self.meta_for_long:
            print("### WARNING: meta instruction is only used for long answer now ###")

        self.use_7132k = use_7132k

        print("### WARNING: Intern v0 version is using. ###")

        self.balance_sample = balance_sample
        if self.balance_sample:
            print('Sample balance is using now!')

        self.mask_human = mask_human
        if self.mask_human:
            print("Only predict answer part")
        else:
            print("predict both human question and assiatant answer")

        self.no_meta = no_meta
        if self.no_meta:
            print("No meta, only used for single data sft")

        print('Loading VIT')
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()
            self.ln_vision.train = disabled_train
            logging.info("freeze vision encoder")
        print('Loading VIT Done')

        self.simple_conv_prompt = simple_conv_prompt
        if self.simple_conv_prompt:
            print("remove long prompt for conversation")

        self.use_qformer = use_qformer
        self.freeze_qformer = freeze_qformer
        self.freeze_qformer_query = freeze_qformer_query
        self.freeze_llama_proj = freeze_llama_proj
        if self.freeze_llama_proj:
            print("Freezing the linear of MiniGPT-4, only used for llama tuning")

        self.instruct_blip = instruct_blip
        if self.instruct_blip:
            print("Tuning in InstructBlip way!!!")

        self.img_embed_unfold = int(img_embed_unfold)
        if self.img_embed_unfold >= 2:
            self.unfold = nn.Unfold((self.img_embed_unfold, self.img_embed_unfold), stride=self.img_embed_unfold)

        if self.use_qformer:
            print('Loading Q-Former')
            self.Qformer, self.query_tokens = self.init_Qformer(
                num_query_token, self.visual_encoder.num_features
            )
            if not self.instruct_blip:
                self.Qformer.bert.embeddings.word_embeddings = None
                self.Qformer.bert.embeddings.position_embeddings = None
                for layer in self.Qformer.bert.encoder.layer:
                    layer.output = None
                    layer.intermediate = None


            else:
                self.Qformer.resize_token_embeddings(len(self.tokenizer))
                old_q_former_model = "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained.pth"
                msg = self.load_from_pretrained(url_or_filename=old_q_former_model)
                print("Load Text FFN for Instruct BLIP")
                print(msg)
            self.Qformer.cls = None

            # msg = self.load_from_pretrained(url_or_filename=q_former_model)
            # print ("Load from BLIP-2")
            # print (msg)

            if self.freeze_qformer and not only_instruct_ft:
                for name, param in self.Qformer.named_parameters():
                    param.requires_grad = False
                self.Qformer = self.Qformer.eval()
                self.Qformer.train = disabled_train
                logging.info("freeze Qformer")

            if self.freeze_qformer and only_instruct_ft:
                for name, param in self.Qformer.named_parameters():
                    param.requires_grad = False
                    if '.layer.' in name and ('.output.' in name or '.intermediate.' in name):
                        print(name)
                        param.requires_grad = True
                logging.info("Only Finetune NLP part of Qformer")

            if self.freeze_qformer_query:
                self.query_tokens.requires_grad = False
                logging.info("freeze Qformer query")
            print('Loading Q-Former Done')

        print('Loading LLAMA')

        if self.use_7132k:
            self.llama_tokenizer = PyTokenizer.from_pretrained("/mnt/petrelfs/share_data/yanhang/tokenizes/V7.model")
            self.flag_image_start = nn.Parameter(torch.zeros([1, 1, 4096]))
            self.flag_image_end = nn.Parameter(torch.zeros([1, 1, 4096]))
            self.flag_image_start.requires_grad = False
            self.flag_image_end.requires_grad = False

            self.eoh = self.llama_tokenizer.decode(torch.Tensor([103027]), skip_special_tokens=True)
            self.eoa = self.llama_tokenizer.decode(torch.Tensor([103028]), skip_special_tokens=True)
            self.itg_id = len(self.llama_tokenizer)
            self.itg_token = '<ITG_TOKEN>'
            self.soi_id = self.itg_id + 1
            self.soi_token = '<SOI_TOKEN>'
            if quant_pretrain:
                new_tokens = [self.itg_token, self.soi_token]
                self.llama_tokenizer.add_tokens(new_tokens)

        else:
            self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
            self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        self.gradient_checkpointing = gradient_checkpointing

        self.llama_lora = False
        if llama_lora is not None:
            llama_freeze = llama_lora.pop('freeze')
            self.llama_freeze = llama_freeze
            setattr(LlamaForCausalLM, 'lora_cfg', llama_lora)
            self.llama_lora = True

        # if using "AutoTokenizer", model can output chinese
        # self.llama_tokenizer = AutoTokenizer.from_pretrained(llama_model, use_fast=False)
        # self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        if self.use_7132k:
            llm_cfg = LlamaConfig(vocab_size=103168, num_hidden_layers=32, num_attention_heads=32,
                                  hidden_size=4096, intermediate_size=11008, rms_norm_eps=1e-5,
                                  intern_converted_llm=True,
                                  bos_token_id=1, eos_token_id=2, pad_token_id=-1,
                                  # sp_id = self.soi_id,
                                  kqvo_bias=True,
                                  )
        else:
            llm_cfg = None

        if fully_open_llm:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                llm_cfg,
                torch_dtype=torch.float32,
            )
        elif self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                llm_cfg,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map="auto"
            )
        elif self.llama_lora:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                llm_cfg,
                torch_dtype=torch.float16,
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                llm_cfg,
                torch_dtype=torch.float16,
            )

        if not fully_open_llm:
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
                if self.llama_lora and name.find("lora") != -1 and not self.llama_freeze:
                    param.requires_grad = True
                # elif name.find("embed_tokens") != -1 or name.find("lm_head") != -1:
                #    param.requires_grad = True
            for name, module in self.llama_model.named_modules():
                module.half()
                if self.llama_lora:
                    # if name.find("lora") != -1 and not self.llama_freeze:
                    #    module.float()
                    if name.find("lora") != -1:
                        module.float()
                    # elif name.find("embed_tokens") != -1 or name.find("lm_head") != -1:
                    #    module.float()
            for name, param in self.llama_model.named_parameters():
                print(name, param.dtype)

        if self.gradient_checkpointing:
            self.llama_model.apply(
                partial(self.llama_model._set_gradient_checkpointing, value=True))

        print('Loading LLAMA Done')

        if self.use_qformer:
            self.llama_proj = nn.Linear(
                self.Qformer.config.hidden_size, self.llama_model.config.hidden_size
            )
            if self.freeze_llama_proj:
                for name, param in self.llama_proj.named_parameters():
                    print("freeze {} for lora only finetuning".format(name))
                    param.requires_grad = False

        else:
            self.llama_proj = nn.Linear(
                self.visual_encoder.embed_dim * self.img_embed_unfold * self.img_embed_unfold,
                self.llama_model.config.hidden_size
            )
            self.llama_proj_extra = nn.Linear(
                self.visual_encoder.embed_dim, self.llama_model.config.hidden_size
            )
        self.max_txt_len = max_txt_len
        self.temp_max_length = self.max_txt_len
        self.end_sym = end_sym

        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<ImageHere>" in raw_prompt]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        else:
            self.prompt_list = []

        temp_long_prompts = [
            '<Img><ImageHere></Img> Describe this image in detail.',
            '<Img><ImageHere></Img> Write a short description for the image.',
            '<Img><ImageHere></Img> Please provide a detailed description of the picture.',
            '<Img><ImageHere></Img> Can you explain what you see in the image in detail?',
            '<Img><ImageHere></Img> Provide a long description of what is presented in the photo.',
        ]

        self.long_prompt_list = [prompt_template.format(p) for p in temp_long_prompts]
        print('Load {} training long  prompts'.format(len(self.long_prompt_list)))
        print('Long Prompt Example \n{}'.format(random.choice(self.long_prompt_list)))

        stop_words_ids = [
            torch.tensor([103028]).to(self.device),
            torch.tensor([103027]).to(self.device),
            torch.tensor([self.soi_id]).to(self.device),
        ]
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

        self.meta_list = [
            'pure_text',
            'pure_conversation',
            'conversation',
            'cc_sub',
        ]

        self.long_list = [
            'pure_text',
            'pure_conversation',
        ]

        self.debug_flag = 1

    def vit_to_cpu(self):
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()

    def encode_img(self, image, instruction=None):
        device = image.device
        # if self.low_resource:
        #    self.vit_to_cpu()
        #    image = image.to("cpu")

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            if self.use_qformer:
                if self.instruct_blip and instruction is not None:
                    if type(instruction) != type([]):
                        instruction = [instruction for i in range(image_embeds.shape[0])]

                    if self.debug_flag:
                        print(instruction)
                    text_Qformer = self.tokenizer(
                        instruction,
                        padding='longest',
                        truncation=True,
                        max_length=50,
                        return_tensors="pt",
                    ).to(image.device)
                    query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
                    Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)
                    query_output = self.Qformer.bert(
                        text_Qformer.input_ids,
                        attention_mask=Qformer_atts,
                        query_embeds=query_tokens,
                        encoder_hidden_states=image_embeds,
                        encoder_attention_mask=image_atts,
                        return_dict=True,
                    )
                else:
                    query_output = self.Qformer.bert(
                        query_embeds=query_tokens,
                        encoder_hidden_states=image_embeds,
                        encoder_attention_mask=image_atts,
                        return_dict=True,
                    )
                inputs_llama = self.llama_proj(query_output.last_hidden_state)
            else:
                num_patches = self.visual_encoder.patch_embed.num_patches
                extra_embed_num = image_embeds.size(1) - num_patches
                extra_embeds = image_embeds[:, :extra_embed_num, :]
                patch_embeds = image_embeds[:, extra_embed_num:, :]

                B, L, D = patch_embeds.size()
                H = int(math.sqrt(L))
                patch_embeds_ = patch_embeds.permute(0, 2, 1).view(B, D, H, H).contiguous()
                patch_embeds_unfold = self.unfold(patch_embeds_)
                patch_embeds_unfold = patch_embeds_unfold.permute(0, 2, 1).contiguous()

                inputs_llama_patch = self.llama_proj(patch_embeds_unfold)
                inputs_llama_extra = self.llama_proj_extra(extra_embeds)

                inputs_llama = torch.cat([inputs_llama_extra, inputs_llama_patch], dim=1)

            if self.use_7132k:  # pad image start/end token embedding
                inputs_llama = torch.cat([self.flag_image_start.expand(inputs_llama.shape[0], -1, -1), inputs_llama,
                                          self.flag_image_end.expand(inputs_llama.shape[0], -1, -1)], dim=1)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
        return inputs_llama, atts_llama

    def interleav_wrap(self, img_embeds, atts_img, text):
        batch_size = img_embeds.shape[0]
        text = text[0]
        text = text.replace('<Img>', '')
        text = text.replace('</Img>', '')
        parts = text.split('<ImageHere>')
        assert batch_size + 1 == len(parts)
        warp_tokens = []
        warp_embeds = []
        warp_attns = []
        soi = (torch.ones([1, 1]) * self.soi_id).long().to(img_embeds.device)
        soi_embeds = self.llama_model.model.embed_tokens(soi)
        temp_len = 0

        for idx, part in enumerate(parts):
            if len(part) > 0:
                part_tokens = self.llama_tokenizer(
                    part, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
                part_embeds = self.llama_model.model.embed_tokens(part_tokens.input_ids)

                warp_tokens.append(part_tokens.input_ids)
                warp_embeds.append(part_embeds)
                temp_len += part_embeds.shape[1]
            if idx < batch_size:
                warp_tokens.append(soi.expand(-1, img_embeds[idx].shape[0] + 1))
                warp_embeds.append(soi_embeds)  ### 1, 1, C
                warp_embeds.append(img_embeds[idx].unsqueeze(0))  ### 1, 34, C
                temp_len += 35

            if temp_len > self.max_txt_len:
                break

        warp_tokens = torch.cat(warp_tokens, dim=1)
        warp_embeds = torch.cat(warp_embeds, dim=1)

        wrapped_target = self.mask_human_targets(warp_tokens).to(img_embeds.device)
        wrapped_atts_img = atts_img[:1, :1].expand(-1, wrapped_target.shape[1]).to(img_embeds.device)
        # print (wrapped_target.shape)
        # print (wrapped_atts_img.shape)

        return warp_embeds[:, :self.max_txt_len].to(img_embeds.device), wrapped_atts_img[:, :self.max_txt_len].to(
            img_embeds.device), wrapped_target[:, :self.max_txt_len].to(img_embeds.device)

    def prompt_wrap(self, img_embeds, atts_img, prompt):
        if prompt:
            batch_size = img_embeds.shape[0]
            p_before, p_after = prompt.split('<ImageHere>')
            if self.use_7132k:
                p_before = p_before.replace('<Img>', '')
                p_after = p_after.replace('</Img>', '')
            p_before_tokens = self.llama_tokenizer(
                p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)

            p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds], dim=1)

            wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])

            if self.mask_human:
                wrapped_target = torch.ones(batch_size, wrapped_img_embeds.shape[1], dtype=torch.long).to(
                    img_embeds.device) * -100
            else:
                target_before = p_before_tokens.input_ids.masked_fill(
                    p_before_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
                ).expand(img_embeds.shape[0], -1)
                target_img = torch.ones(img_embeds.shape[0], img_embeds.shape[1], dtype=torch.long).to(
                    img_embeds.device) * -100
                wrapped_target = torch.cat([target_before, target_img], dim=1)

            return wrapped_img_embeds, wrapped_atts_img, wrapped_target
        else:
            return img_embeds, atts_img

    def mask_human_targets(self, input_ids, pure=False):
        target_batch = []
        for bs in range(input_ids.shape[0]):
            cur_idx = 0
            ids = input_ids[bs]
            targets = copy.deepcopy(ids)
            last_eoa = 0
            last_eoh = 0
            for i, temp_id in enumerate(ids):
                if temp_id == 103027:  #### end of human
                    targets[cur_idx:i + 6] = -100
                    cur_idx = i + 6
                    last_eoh = i
                elif temp_id == 103028:  ### end of assistant
                    cur_idx = i + 1
                    last_eoa = i
                elif temp_id == 1:  ### eos and following pad
                    targets[i + 1:] = -100  #### loss on eos, but not on pad
                    break
                elif temp_id == self.soi_id and targets[i] != -100 and not pure:  ### start of image
                    targets[i + 1: i + 35] = -100
            if temp_id != 1 and last_eoa > last_eoh:  ### trunction, end at last question
                targets[last_eoa + 1:] = -100  #### mask all after the last answer

            target_batch.append(targets.unsqueeze(0))
            if self.debug_flag:
                targets_vis = targets.clone()
                targets_vis[targets_vis == -100] = 0
                targets_vis_tokens = self.llama_tokenizer.convert_ids_to_tokens(targets_vis)
                print(''.join(self.llama_tokenizer.convert_ids_to_tokens(ids)))
                print('-----------')
                print(''.join(targets_vis_tokens))
                print('-----------------------------')
        target_batch = torch.cat(target_batch, dim=0)
        return target_batch

    def extract_instruct(self, text_input):
        text_output = []
        for text in text_input:
            end = text.find('### Assistant')
            if end != -1:
                text = text[:end]
                if text.startswith('### Human:'):
                    text = text[10:]
            text_output.append(text)
        return text_output

    def align_text(self, samples):
        text_new = []
        text = [t + self.eoa + ' </s>' for t in samples["text_input"]]
        for i in range(len(text)):
            temp = text[i]
            temp = temp.replace('###Human', '<|User|>')
            temp = temp.replace('### Human', '<|User|>')
            temp = temp.replace('<|User|> :', '<|User|>:')
            temp = temp.replace('<|User|>: ', '<|User|>:')
            temp = temp.replace('<|User|>', ' <|User|>')

            temp = temp.replace('###Assistant', '<|Bot|>')
            temp = temp.replace('### Assistant', '<|Bot|>')
            temp = temp.replace('<|Bot|> :', '<|Bot|>:')
            temp = temp.replace('<|Bot|>: ', '<|Bot|>:')
            temp = temp.replace('<|Bot|>', self.eoh + ' <|Bot|>')
            if temp.find('<|User|>') > temp.find('<|Bot|>'):
                temp = temp.replace(' <|User|>', self.eoa + ' <|User|>')
            text_new.append(temp)
            # print (temp)
        return text_new

    def text2emb(self, text, data_type):
        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.temp_max_length,
            add_special_tokens=False
        ).to(self.device)

        if self.mask_human:
            targets = self.mask_human_targets(to_regress_tokens.input_ids)
            targets = targets.to(self.device)
        #### mask human for short data, they share similar pattern in most cases
        elif self.meta_for_long and data_type not in self.meta_list:
            targets = self.mask_human_targets(to_regress_tokens.input_ids)
            targets = targets.to(self.device)
        else:
            targets = to_regress_tokens.input_ids.masked_fill(
                to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
            )
            for i, temp in enumerate(targets):
                for j, t in enumerate(temp):
                    if t == -100:
                        targets[i][j] = 1
                        break
            if self.debug_flag:
                targets_vis = targets.clone()[0]
                targets_vis[targets_vis == -100] = 0
                targets_vis_tokens = self.llama_tokenizer.convert_ids_to_tokens(targets_vis)
                print(''.join(self.llama_tokenizer.convert_ids_to_tokens(to_regress_tokens.input_ids[0])))
                print('-----------')
                print(''.join(targets_vis_tokens))
                print('-----------------------------')

        return to_regress_tokens, targets

    def forward(self, samples):
        if self.debug_flag:
            self.debug_flag += 1
        if self.debug_flag > 20:
            self.debug_flag = 0

        self.llama_tokenizer.padding_side = "right"

        meta_instruction = """meta instruction
You are an AI assistant whose name is 浦语.
- 浦语 is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- 浦语 can understand and communicate fluently in the language chosen by the user such as English and 中文.
conversation
"""

        if self.balance_sample:
            if samples['data_type'][0] in self.long_list:
                self.temp_max_length = 1024
                new_samples = {}
                for key in list(samples):
                    if type(samples[key]) == type([]):
                        new_samples[key] = samples[key][:2]
                new_samples['image'] = samples["image"][:2]
                samples = new_samples

            else:
                self.temp_max_length = self.max_txt_len

        image = samples["image"]
        if len(image.shape) == 5:
            assert image.shape[0] == 1
            image = image.squeeze(0)
        batch_size = image.shape[0]

        # print (meta_instruction)
        if samples['data_type'][0] == 'pure_text':
            # print('-----------pure_text:------------\n', samples['prompt'][0] + samples["text_input"][0])
            prompt = ''
            img_embeds, atts_img = self.encode_img(image)

            for i in range(len(samples['prompt'])):
                temp = samples['prompt'][i]
                temp = temp.replace('###Human', '<|User|>')
                temp = temp.replace('### Human', '<|User|>')
                temp = temp.replace('<|User|> :', '<|User|>:')
                temp = temp.replace('<|User|>: ', '<|User|>:')
                temp = temp.replace('<|User|>', ' <|User|>:')

                temp = temp.replace('###Assistant', '<|Bot|>')
                temp = temp.replace('### Assistant', '<|Bot|>')
                temp = temp.replace('<|Bot|> :', '<|Bot|>:')
                temp = temp.replace('<|Bot|>: ', '<|Bot|>:')
                temp = temp.replace('<|Bot|>', self.eoh + ' <|Bot|>')
                # print(temp)
                samples['prompt'][i] = temp

            prompt_tokens = self.llama_tokenizer(
                samples['prompt'],
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_txt_len,
                padding=True).to(image.device)
            img_embeds = self.llama_model.model.embed_tokens(prompt_tokens.input_ids).expand(batch_size, -1,
                                                                                             -1) + img_embeds.sum() * 0  ##pure text instruction
            atts_img = torch.ones(batch_size, img_embeds.shape[1], dtype=torch.long).to(image.device)
            if self.mask_human:
                wrapped_target = torch.ones(batch_size, img_embeds.shape[1], dtype=torch.long).to(image.device) * -100
            else:
                wrapped_target = prompt_tokens.input_ids.masked_fill(
                    prompt_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
                )

            text = self.align_text(samples)
            to_regress_tokens, targets = self.text2emb(text, samples['data_type'][0])

            to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
            img_embeds = torch.cat([img_embeds, to_regress_embeds], dim=1)
            atts_img = torch.cat([atts_img, to_regress_tokens.attention_mask], dim=1)
            wrapped_target = torch.cat([wrapped_target, targets], dim=1)



        # conversation: share gpt --> multi round conversation without image
        elif samples['data_type'][0] == 'pure_conversation':
            prompt = " <|User|>:"

            batch_size = image.shape[0]
            img_embeds, atts_img = self.encode_img(image)
            prompt_tokens = self.llama_tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_txt_len,
                padding=True).to(image.device)

            img_embeds = self.llama_model.model.embed_tokens(prompt_tokens.input_ids).expand(batch_size, -1,
                                                                                             -1) + img_embeds.sum() * 0  ##pure text instruction
            atts_img = torch.ones(batch_size, img_embeds.shape[1], dtype=torch.long).to(image.device)
            if self.mask_human:
                wrapped_target = torch.ones(batch_size, img_embeds.shape[1], dtype=torch.long).to(image.device) * -100
            else:
                wrapped_target = prompt_tokens.input_ids.masked_fill(
                    prompt_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
                ).expand(batch_size, -1)

            text = self.align_text(samples)
            to_regress_tokens, targets = self.text2emb(text, samples['data_type'][0])

            to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
            img_embeds = torch.cat([img_embeds, to_regress_embeds], dim=1)
            atts_img = torch.cat([atts_img, to_regress_tokens.attention_mask], dim=1)
            wrapped_target = torch.cat([wrapped_target, targets], dim=1)

        # conversation: LLAVA-150K
        elif samples['data_type'][0] == 'conversation' or samples['data_type'][0] == 'brief_conversation':
            prompt = ' <|User|>:<Img><ImageHere></Img>'
            instrct = 'Describe this image in detail.'
            img_embeds, atts_img = self.encode_img(image, instrct)
            img_embeds, atts_img, wrapped_target = self.prompt_wrap(img_embeds, atts_img, prompt)

            text = self.align_text(samples)
            to_regress_tokens, targets = self.text2emb(text, samples['data_type'][0])

            to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
            img_embeds = torch.cat([img_embeds, to_regress_embeds], dim=1)
            atts_img = torch.cat([atts_img, to_regress_tokens.attention_mask], dim=1)
            wrapped_target = torch.cat([wrapped_target, targets], dim=1)

        # VQA: ScienceQA
        elif 'vqa' in samples['data_type'][0]:  #### now science qa is 'long_vqa' type
            prompt = ' <|User|>:<Img><ImageHere></Img>'
            instrct = samples['question']
            if self.debug_flag:
                print(samples['question'])
            # for i in range(len(samples['text_input'])):
            #    temp = samples['text_input'][i]
            #    temp = temp.split('###')[0]
            #    instrct.append(samples['question'][i])

            img_embeds, atts_img = self.encode_img(image, instrct)
            img_embeds, atts_img, wrapped_target = self.prompt_wrap(img_embeds, atts_img, prompt)

            text = self.align_text(samples)
            to_regress_tokens, targets = self.text2emb(text, samples['data_type'][0])

            to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
            img_embeds = torch.cat([img_embeds, to_regress_embeds], dim=1)
            atts_img = torch.cat([atts_img, to_regress_tokens.attention_mask], dim=1)
            wrapped_target = torch.cat([wrapped_target, targets], dim=1)

        #### cc_sbu, long caption
        elif samples['data_type'][0] == 'cc_sbu':
            prompt = random.choice(self.long_prompt_list).split('</Img>')[-1]
            #### move the instruction to input for simplfy
            for i in range(batch_size):
                temp = prompt + samples['text_input'][i]
                samples['text_input'][i] = temp

            instrct = prompt.split('###')[0]
            prompt = ' <|User|>:<Img><ImageHere></Img>'
            img_embeds, atts_img = self.encode_img(image, instrct)
            img_embeds, atts_img, wrapped_target = self.prompt_wrap(img_embeds, atts_img, prompt)

            text = self.align_text(samples)
            to_regress_tokens, targets = self.text2emb(text, samples['data_type'][0])

            to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
            img_embeds = torch.cat([img_embeds, to_regress_embeds], dim=1)
            atts_img = torch.cat([atts_img, to_regress_tokens.attention_mask], dim=1)
            wrapped_target = torch.cat([wrapped_target, targets], dim=1)

        elif samples['data_type'][0] == "vigc":
            for i in range(batch_size):
                prompt = samples["prompt"][i]
                text_input = samples["text_input"][i]
                temp = f"{prompt} \n### Assistant: {text_input}"
                samples["text_input"][i] = temp

            img_embeds, atts_img = self.encode_img(image, samples["prompt"])
            prompt = ' <|User|>:<Img><ImageHere></Img>'
            img_embeds, atts_img, wrapped_target = self.prompt_wrap(img_embeds, atts_img, prompt)
            text = self.align_text(samples)
            to_regress_tokens, targets = self.text2emb(text, samples['data_type'][0])

            to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
            img_embeds = torch.cat([img_embeds, to_regress_embeds], dim=1)
            atts_img = torch.cat([atts_img, to_regress_tokens.attention_mask], dim=1)
            wrapped_target = torch.cat([wrapped_target, targets], dim=1)

        #### zhihu interleaved data
        elif samples['data_type'][0] == 'interleav':
            img_embeds, atts_img = self.encode_img(image)
            text = self.align_text(samples)
            img_embeds, atts_img, wrapped_target = self.interleav_wrap(img_embeds, atts_img, text)

        #### zhihu interleaved data with text only
        elif samples['data_type'][0] == 'pure_interleav':
            text = self.align_text(samples)
            text_tokens = self.llama_tokenizer(
                text, return_tensors="pt", add_special_tokens=False).to(self.device)
            text_embeds = self.llama_model.model.embed_tokens(text_tokens.input_ids)

            img_embeds = text_embeds[:, :self.max_txt_len]
            atts_img = text_tokens.attention_mask[:, :self.max_txt_len]
            targets = self.mask_human_targets(text_tokens.input_ids, pure=True)
            targets = targets.to(image.device)
            wrapped_target = targets[:, :self.max_txt_len]

        # others: short caption
        elif self.prompt_list:
            prompt = random.choice(self.prompt_list).split('</Img>')[-1]
            instrct = prompt.split('###')[0]
            #### move the instruction to input for simplfy
            for i in range(batch_size):
                temp = prompt + samples['text_input'][i]
                samples['text_input'][i] = temp

            text = self.align_text(samples)
            prompt = ' <|User|>:<Img><ImageHere></Img>'
            img_embeds, atts_img = self.encode_img(image, instrct)
            img_embeds, atts_img, wrapped_target = self.prompt_wrap(img_embeds, atts_img, prompt)

            text = self.align_text(samples)
            to_regress_tokens, targets = self.text2emb(text, samples['data_type'][0])

            to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
            img_embeds = torch.cat([img_embeds, to_regress_embeds], dim=1)
            atts_img = torch.cat([atts_img, to_regress_tokens.attention_mask], dim=1)
            wrapped_target = torch.cat([wrapped_target, targets], dim=1)

        batch_size = img_embeds.shape[0]

        if self.meta_for_long and samples['data_type'][0] in self.meta_list:
            meta_tokens = self.llama_tokenizer(
                meta_instruction,
                return_tensors="pt",
                padding="longest",
                truncation=False,
                add_special_tokens=True
            ).to(image.device)
            meta_embeds = self.llama_model.model.embed_tokens(meta_tokens.input_ids).expand(batch_size, -1, -1)
            atts_meta = meta_tokens.attention_mask.expand(batch_size, -1)
            meta_target = torch.ones(meta_embeds.shape[0], meta_embeds.shape[1], dtype=torch.long).to(
                image.device) * -100

        else:
            bos = torch.ones([batch_size, 1]) * self.llama_tokenizer.bos_token_id
            bos = bos.long().to(image.device)
            meta_embeds = self.llama_model.model.embed_tokens(bos)
            atts_meta = atts_img[:, :1]
            meta_target = torch.ones(meta_embeds.shape[0], 1, dtype=torch.long).to(image.device) * -100

        if self.debug_flag:
            le = len(samples['text_input'])
            sh = targets.shape
            data_type = samples['data_type'][0]
            print(
                f'DataType: {data_type}. Current max length: {self.temp_max_length}, BatchSize is {le}, Sample shape is {sh}')

        ###
        targets = torch.cat([meta_target, wrapped_target], dim=1)
        inputs_embeds = torch.cat([meta_embeds, img_embeds], dim=1)
        attention_mask = torch.cat([atts_meta, atts_img], dim=1)

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss
        return {"loss": loss}
        '''
        with self.maybe_autocast():
            loss, loss_o, loss_l = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )

        return {
                 "loss": loss,
                 "loss_o": loss_o,
                 "loss_l": loss_l,
               }
        '''

    def caption_generate(
            self,
            samples,
            use_nucleus_sampling=False,
            num_beams=5,
            max_length=256,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.5,
            length_penalty=1,
            num_captions=1,
            temperature=1,
    ):

        image = samples["image"].to(self.device)
        txt_prompt = samples["prompt"]

        self.use_meta = True
        if self.use_meta:
            img_prompt = meta_instruction + ' <|Human|>:<ImageHere> '
        else:
            img_prompt = ' <|Human|>:<ImageHere> '

        prompts = [img_prompt + _ + self.eoh + ' <|Assistant|>:' for _ in txt_prompt]

        img_embeds, _ = self.encode_img(image)

        prompt_segs = [prompt.split('<ImageHere>') for prompt in prompts]
        prompt_seg_tokens = [[
            self.llama_tokenizer(seg,
                                 return_tensors='pt',
                                 add_special_tokens=i == 0).
                to(self.llama_model.model.embed_tokens.weight.device).input_ids
            for i, seg in enumerate(prompt_seg)
        ] for prompt_seg in prompt_segs]

        prompt_seg_embs = [[
            self.llama_model.model.embed_tokens(seg)
            for seg in prompt_seg_token
        ] for prompt_seg_token in prompt_seg_tokens]

        first_prompt_seg_embeds = [
            prompt_seg_embed[0] for prompt_seg_embed in prompt_seg_embs
        ]
        first_prompt_seg_embeds = torch.cat(first_prompt_seg_embeds, dim=0)
        second_prompt_seg_embeds = [
            prompt_seg_embed[1] for prompt_seg_embed in prompt_seg_embs
        ]
        second_prompt_seg_embeds = torch.cat(second_prompt_seg_embeds, dim=0)

        prompt_seg_embs = [
            first_prompt_seg_embeds, img_embeds, second_prompt_seg_embeds
        ]
        prompt_embs = torch.cat(prompt_seg_embs, dim=1)

        # generate output
        outputs = self.llama_model.generate(
            inputs_embeds=prompt_embs,
            max_length=max_length,
            num_beams=num_beams,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
            eos_token_id=self.llama_tokenizer.eos_token_id,
            num_return_sequences=num_captions,
            do_sample=use_nucleus_sampling,
        )
        res = []
        for i, output in enumerate(outputs):
            if output[0] == 0:
                output = output[1:]
            if output[0] == 1:
                output = output[1:]
            output_text = self.llama_tokenizer.decode(output,
                                                      add_special_tokens=False)
            output_text = output_text.split(self.eoa)[0]
            output_text = output_text.split('<|Assistant|>')[-1].strip()
            print(output_text)
            res.append(output_text)
        return res

    @torch.no_grad()
    def generate(
            self,
            samples,
            use_nucleus_sampling=False,
            num_beams=5,
            max_length=256,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.5,
            length_penalty=1,
            num_captions=1,
            temperature=1,
            refs=None,
    ):

        # text = self.align_text(samples)
        text = samples['text_input']
        if 'image' in samples.key():
            image = samples['image']
            img_embeds, atts_img = self.encode_img(image)
            img_embeds, atts_img, wrapped_target = self.interleav_wrap(img_embeds, atts_img, text)
        else:  ### only text input
            input_tokens = self.llama_tokenizer(
                text, return_tensors="pt", add_special_tokens=False).to(self.device)
            img_embeds = self.llama_model.model.embed_tokens(input_tokens.input_ids)
            atts_img = torch.ones(1, 1).to(self.device)
        if 'previous_fea' in samples.key():
            previous_fea = samples['previous_fea']
            img_embeds = torch.cat([previous_fea, img_embeds], dim=1)

        atts_mask = atts_img[:1, :1].expand(-1, img_embeds.shape[1]).to(img_embeds.device)
        with self.maybe_autocast():
            outputs = self.llama_model.generate(
                inputs_embeds=img_embeds,
                attention_mask=atts_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
                stopping_criteria=self.stopping_criteria,
            )
        output_token = outputs[0]
        if output_token[0] == 0:
            output_token = output_token[1:]
        output_text = self.llama_tokenizer.decode(output_token, add_special_tokens=False)
        print(output_text)
        print(output_token)
        if output_token[-1] == self.soi_id:  ### get image here
            samples['previous_fea'] = img_embeds
            if refs is not None:
                image = refs.find_ref(output_text[-60:-1]).to(img_embeds.device)
            else:
                print('Warning, not referce image, use emepty image')
                image = torch.zeros(1, 3, 224, 224).to(img_embeds.device)
            samples['image'] = image
            output_text = output_text.replace('<IMG>', '<Img><ImageHere></Img>')
            samples['text_input'] = output_text
            samples['text_full'] = samples['text_full'] + output_text

            samples = self.generate(
                samples,
                use_nucleus_sampling,
                num_beams,
                max_length,
                min_length,
                top_p,
                repetition_penalty,
                length_penalty,
                num_captions,
                temperature,
                refs=refs,
            )
        else:
            samples['text_full'] = samples['text_full'] + output_text
        return samples

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        q_former_model = cfg.get("q_former_model",
                                 "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llama_model = cfg.get("llama_model")
        llama_lora = cfg.get('llama_lora', None)
        fully_open_llm = cfg.get("fully_open_llm", False)

        if llama_lora is not None and llama_lora.get('learn_param', None) is None:
            llama_lora.learn_param = ['q', 'k']

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        use_qformer = cfg.get("use_qformer", True)
        img_embed_unfold = cfg.get("img_embed_unfold", 1)
        freeze_qformer = cfg.get("freeze_qformer", True)
        freeze_qformer_query = cfg.get("freeze_qformer_query", True)
        low_resource = cfg.get("low_resource", False)
        gradient_checkpointing = cfg.get("gradient_checkpointing", False)
        freeze_llama_proj = cfg.get("freeze_llama_proj", False)

        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        simple_conv_prompt = cfg.get("simple_conv_prompt", False)
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')
        cfg_7132k = cfg.get("7132k", '')
        use_7132k = cfg_7132k != ''
        mask_human = cfg.get("mask_human", True)
        data_debug = cfg.get('data_debug', False)
        meta_for_long = cfg.get('meta_for_long', False)
        only_load_qformer = cfg.get('only_load_qformer', False)
        no_meta = cfg.get('no_meta', False)
        balance_sample = cfg.get('balance_sample', False)

        instruct_blip = cfg.get("instruct_blip", False)
        only_instruct_ft = cfg.get("only_instruct_ft", False)
        quant_pretrain = cfg.get('quant_pretrain', False)

        merge_lora = cfg.get('merge_lora', False)
        merge_lora_scale = cfg.get('merge_lora_scale', 2)

        model = cls(
            vit_model=vit_model,
            q_former_model=q_former_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            freeze_qformer=freeze_qformer,
            freeze_qformer_query=freeze_qformer_query,
            instruct_blip=instruct_blip,
            use_qformer=use_qformer,
            img_embed_unfold=img_embed_unfold,
            num_query_token=num_query_token,
            llama_model=llama_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            low_resource=low_resource,
            end_sym=end_sym,
            fully_open_llm=fully_open_llm,
            llama_lora=llama_lora,
            gradient_checkpointing=gradient_checkpointing,
            freeze_llama_proj=freeze_llama_proj,
            use_7132k=use_7132k,
            simple_conv_prompt=simple_conv_prompt,
            mask_human=mask_human,
            meta_for_long=meta_for_long,
            no_meta=no_meta,
            balance_sample=balance_sample,
            only_instruct_ft=only_instruct_ft,
            quant_pretrain=quant_pretrain,
        )

        if data_debug:
            print("#### Warning! Dataset debug, skip model loading. ####")
            return model

        if use_7132k:
            if quant_pretrain:
                model.llama_model.resize_token_embeddings(len(model.llama_tokenizer))
            ckpt_path = cfg['7132k'].get("ckpt", "")  # load weights of 7132k
            if ckpt_path:
                print("Load 7132k-LLM Checkpoint: {}".format(ckpt_path))
                ckpt = torch.load(ckpt_path, map_location="cpu")
                for name, param in model.llama_model.named_parameters():
                    if 'lora_B' in name:
                        try:
                            assert torch.mean(param) == 0  ### check the lora is init mode
                        except:
                            print("Lora init with wrong weight, init again")
                    if 'lora_A' in name:
                        print(name, torch.mean(param))
                        nn.init.kaiming_uniform_(param)
                    if 'lora_B' in name:
                        print(name, torch.mean(param))
                        nn.init.zeros_(param)
                if only_load_qformer:
                    print("### WAINGING, only qformer is loaded from pretrian model, lora is removed! ###")
                    keys = ckpt.keys()
                    for key in list(keys):
                        if 'lora' in key:
                            _ = ckpt.pop(key)
                            print(f"Remove {key} from pretrain model")

                if merge_lora:
                    model = merge_lora_func(model, ckpt, scale=merge_lora_scale, path='ckpt_path')

                else:
                    msg = model.load_state_dict(ckpt, strict=False)
                    print(msg)
                    print('------------end load 7132k-LLM------------------')

                    if only_load_qformer:
                        for name, param in model.llama_model.named_parameters():
                            if 'lora_B' in name:
                                assert torch.mean(param) == 0  ### check the lora is init mode

        for name, param in model.llama_model.named_parameters():
            if name.find("embed_tokens") != -1 or name.find("lm_head") != -1:
                param.requires_grad = False

        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            print("Load BLIP2-LLM Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)
            print(msg)
            print('------------end MLLM------------------')

        return model
