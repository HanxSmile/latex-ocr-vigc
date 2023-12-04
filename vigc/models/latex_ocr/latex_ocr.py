import torch
import re

import vigc.models.latex_ocr.vit as vit
import vigc.models.latex_ocr.hybrid as hybrid
import vigc.models.latex_ocr.transformer as transformer

from transformers import PreTrainedTokenizerFast
from vigc.common.registry import registry
from vigc.models.blip2_models.blip2 import Blip2Base


@registry.register_model("latex_ocr")
class LatexOCRModel(Blip2Base):
    """
    Nougat model for formula recognition.
    Supported model types:
        - default
    Usage:
        >>> from vigc.models import load_model
        >>> model = load_model("latex_ocr", "default")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "default": "configs/models/latex_ocr.yaml",
        "hybrid": "configs/models/hybrid_latex_ocr.yaml"
    }

    pad_token = "[PAD]"
    bos_token = "[BOS]"
    eos_token = "[EOS]"
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def __init__(
            self,
            *,
            encoder_structure,
            encoder_args,
            decoder_args,
            tokenizer
    ):
        super().__init__()
        if encoder_structure.lower() == 'vit':
            self.encoder = vit.get_encoder(encoder_args)
        elif encoder_structure.lower() == 'hybrid':
            self.encoder = hybrid.get_encoder(encoder_args)
        else:
            raise NotImplementedError('Encoder structure "%s" not supported.' % encoder_structure)
        self.decoder = transformer.get_decoder(decoder_args)
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer)
        self.tokenizer.add_special_tokens({'pad_token': self.pad_token})
        self.tokenizer.add_special_tokens({'bos_token': self.bos_token})
        self.tokenizer.add_special_tokens({'eos_token': self.eos_token})
        self.tokenizer.pad_token_id = self.pad_token_id
        self.tokenizer.bos_token_id = self.bos_token_id
        self.tokenizer.eos_token_id = self.eos_token_id
        self.max_seq_len = decoder_args.max_seq_len

    def forward(self, samples):
        image, text = samples["image"], samples["text_input"]

        text_inputs = self.tokenize(text).to(image.device)
        tgt_seq, tgt_mask = text_inputs["input_ids"], text_inputs["attention_mask"].bool()
        with self.maybe_autocast():
            encoded = self.encoder(image)
            loss = self.decoder(tgt_seq, context=encoded, mask=tgt_mask)
        return {"loss": loss}

    def tokenize(self, texts):
        text_inputs = self.tokenizer(
            [self.bos_token + _ + self.eos_token for _ in texts],
            return_token_type_ids=False,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_seq_len,
        )
        return text_inputs

    @staticmethod
    def detokenize(tokens, tokenizer):
        toks = [tokenizer.convert_ids_to_tokens(tok) for tok in tokens]
        for b in range(len(toks)):
            for i in reversed(range(len(toks[b]))):
                if toks[b][i] is None:
                    toks[b][i] = ''
                toks[b][i] = toks[b][i].replace('Ġ', ' ').strip()
                if toks[b][i] in (['[BOS]', '[EOS]', '[PAD]']):
                    del toks[b][i]
        return toks

    @staticmethod
    def token2str(tokens, tokenizer) -> list:
        if len(tokens.shape) == 1:
            tokens = tokens[None, :]
        dec = [tokenizer.decode(tok) for tok in tokens]
        return [''.join(detok.split(' ')).replace('Ġ', ' ').replace('[EOS]', '').replace('[BOS]', '').replace('[PAD]',
                                                                                                              '').strip()
                for detok in dec]

    @staticmethod
    def post_process(s: str):
        """Remove unnecessary whitespace from LaTeX code.

        Args:
            s (str): Input string

        Returns:
            str: Processed image
        """
        text_reg = r'(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})'
        letter = '[a-zA-Z]'
        noletter = '[\W_^\d]'
        names = [x[0].replace(' ', '') for x in re.findall(text_reg, s)]
        s = re.sub(text_reg, lambda match: str(names.pop(0)), s)
        news = s
        while True:
            s = news
            news = re.sub(r'(?!\\ )(%s)\s+?(%s)' % (noletter, noletter), r'\1\2', s)
            news = re.sub(r'(?!\\ )(%s)\s+?(%s)' % (noletter, letter), r'\1\2', news)
            news = re.sub(r'(%s)\s+?(%s)' % (letter, noletter), r'\1\2', news)
            if news == s:
                break
        return s

    @torch.no_grad()
    def generate(
            self,
            samples,
            temperature: float = 0.2
    ):

        image = samples["image"]
        bs = image.shape[0]
        with self.maybe_autocast():
            encoded = self.encoder(image)

            outputs = self.decoder.generate(
                (torch.LongTensor([self.bos_token_id] * bs)[:, None]).to(image.device),
                self.max_seq_len,
                eos_token=self.eos_token_id, context=encoded,
                temperature=temperature
            )
        pred_tokens = self.detokenize(outputs, self.tokenizer)
        pred_str = self.token2str(outputs, self.tokenizer)
        pred_str = [self.post_process(_) for _ in pred_str]
        return {"pred_tokens": pred_tokens, "pred_str": pred_str}

    @classmethod
    def from_config(cls, cfg):

        encoder_structure = cfg.get("encoder_structure")
        encoder_args = cfg.get("encoder_args")
        decoder_args = cfg.get("decoder_args")
        tokenizer = cfg.get("tokenizer")

        model = cls(
            encoder_structure=encoder_structure,
            encoder_args=encoder_args,
            decoder_args=decoder_args,
            tokenizer=tokenizer
        )

        model.load_checkpoint_from_config(cfg)

        return model
