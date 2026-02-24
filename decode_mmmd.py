# -*- encoding: utf-8 -*-

import argparse
import json
import re
import typing
from functools import partial
from pprint import pp

import torch
import torch.nn.functional as F
from PIL import Image
from sat.generation.autoregressive_sampling import BaseStrategy, filling_sequence
from sat.generation.sampling_strategies.base_strategy import top_k_logits
from sat.model import AutoModel
from sat.model.mixins import CachedAutoregressiveMixin
from sat.mpu.initialize import (
    get_model_parallel_group,
    get_model_parallel_src_rank,
    get_model_parallel_world_size,
)
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.models.detr.image_processing_detr import defaultdict

from finetune_visualglm import FineTuneVisualGLMModel
from model.blip2 import BlipImageEvalProcessor
from model.chat import get_masks_and_position_ids_glm


class AStrategy:
    def __init__(
        self,
        invalid_slices=[],
        temperature=1.0,
        top_k=200,
        eps=1e-4,
        top_p=0.0,
        repetition_penalty=1.0,
        end_tokens=None,
    ):
        self.repetition_penalty = repetition_penalty
        self.invalid_slices = invalid_slices
        self.temperature = temperature
        self.topk = top_k
        self.top_p = top_p
        self.eps = eps
        if end_tokens is None:
            end_tokens = []
        self.end_tokens = end_tokens
        self._is_done = False
        self.context_length = None
        self.probs = None
        self.preds = None

    @property
    def is_done(self) -> bool:
        return self._is_done

    def forward(self, logits, tokens, mems, temperature=None, nan_default_token=None):
        if self.context_length is None:
            self.context_length = tokens.shape[-1]
        if temperature is None:
            temperature = self.temperature
        if torch.isnan(logits).any():
            if nan_default_token is None:
                raise ValueError(
                    "nan in logits, set nan_default_token to proceed in BaseStrategy.forward."
                )
            logits.fill_(-1000)
            logits[..., nan_default_token] = 0
        # apply repetition penalty
        penalty_mat = torch.ones_like(logits).float()
        if tokens.shape[-1] > self.context_length:
            penalty_mat.scatter_(
                1,
                tokens[:, self.context_length :],
                torch.ones_like(tokens[:, self.context_length :]).float()
                * self.repetition_penalty,
            )
        penalty_mat *= temperature
        logits = logits.float() / penalty_mat

        for invalid_slice in self.invalid_slices:
            logits[..., invalid_slice] = -65504
        logits_all = logits.clone().detach()
        logits = top_k_logits(logits, self.topk, self.top_p)
        probs = F.softmax(logits, dim=-1)  # float is essetial, due to a bug in Pytorch
        pred = torch.multinomial(probs, num_samples=1)
        if get_model_parallel_world_size() > 1:
            torch.distributed.broadcast(
                pred, get_model_parallel_src_rank(), group=get_model_parallel_group()
            )
        if pred.numel() == 1 and pred.item() in self.end_tokens:
            self._is_done = True

        pred = pred.view(tokens.shape[0], 1)
        tokens = torch.cat((tokens, pred), dim=1)

        # save the predicted tokens and probabilities
        assert len(pred) == 1  # batch size == 1
        if self.preds is None:
            self.preds = pred.detach()
        else:
            self.preds = torch.cat((self.preds, pred), dim=1)

        probs_all = F.softmax(logits_all, dim=-1)
        probs_all = probs_all.unsqueeze(1)  # (bsz,vocab) =>(bsz,seq,vocab) seq_len = 1
        if self.probs is None:
            self.probs = probs_all.detach()
        else:
            self.probs = torch.cat((self.probs, probs_all), dim=1)

        return tokens, mems

    def finalize(self, tokens, mems):
        self._is_done = False
        self.context_length = None
        return tokens, mems


def load_mmmd_data(args):
    examples = defaultdict(dict)

    with open(args.test_data) as f:
        for line in f.readlines():
            ex = json.loads(line)
            img, prompt, label = ex["img"], ex["prompt"], ex["label"]
            examples[prompt]["input_image"] = img
            examples[prompt]["prompt"] = prompt
            for line in label.splitlines():
                if not line:
                    continue
                field, value = line.split(": ", maxsplit=1)
                if field == "### 答":
                    field = "\n" + field

                if field == "图片合适":
                    if value == "是":
                        examples[prompt]["pos_image"] = img
                    else:
                        examples[prompt].setdefault("neg_images", []).append(img)
                else:
                    examples[prompt][field] = value

    examples = list(examples.values())
    return examples
    # input_image
    # prompt
    # tasks...
    # pos_image
    # neg_images
    # response


def get_mmmd_model(args):
    # load model
    model, _ = AutoModel.from_pretrained(
        args.from_pretrained,
        args=argparse.Namespace(
            fp16=True,
            skip_init=True,
            use_gpu_initialization=torch.cuda.is_available(),
            device="cuda" if torch.cuda.is_available() else "cpu",
        ),
    )
    model = model.eval()
    model.add_mixin("auto-regressive", CachedAutoregressiveMixin())
    tokenizer = AutoTokenizer.from_pretrained(
        "THUDM/chatglm-6b", trust_remote_code=True
    )
    return model, tokenizer


def mmmd_generate(model, tokenizer, context, image, prefix, end_tokens, args):
    if image is None:
        inputs = tokenizer([context], return_tensors="pt").to(
            model.parameters().__next__().device
        )["input_ids"][0]
        pre_image = 0
        torch_image = None
    else:
        prompt = "<img></img>" + context
        image_position = len("<img>")
        processor = BlipImageEvalProcessor(224)
        image = processor(image.convert("RGB"))
        torch_image = (
            image.unsqueeze(0).to(torch.float16).to(next(model.parameters()).device)
        )
        input0 = tokenizer.encode(prompt[:image_position], add_special_tokens=False)
        input1 = [tokenizer.pad_token_id] * model.image_length
        input2 = tokenizer.encode(prompt[image_position:], add_special_tokens=False)
        inputs = sum([input0, input1, input2], [])  # pre_text, image_tokens, post_text
        inputs = torch.tensor(tokenizer.build_inputs_with_special_tokens(inputs)).to(
            model.parameters().__next__().device
        )
        pre_image = len(input0)

    # ---------------
    # Next, we manually set the format to keep flexibility.
    mask_position = len(inputs) - 2
    context_length = len(inputs) - 1  # all before sop

    if prefix is not None:
        prefix = torch.tensor(
            tokenizer.encode(prefix, add_special_tokens=False), device=inputs.device
        )
        inputs = torch.cat((inputs, prefix))
        prefix_len = len(prefix)
    else:
        prefix_len = 0

    get_func = partial(
        get_masks_and_position_ids_glm,
        mask_position=mask_position,
        context_length=context_length,
    )
    seq = torch.cat(
        [
            inputs,
            torch.tensor([-1] * (args.max_length - len(inputs)), device=inputs.device),
        ],
        dim=0,
    )
    # ---------------
    # from sat.generation.sampling_strategies import BeamSearchStrategy
    # strategy = BeamSearchStrategy(num_beams, length_penalty=1., prefer_min_length=5, end_tokens=[tokenizer.eos_token_id], consider_end=True, no_repeat_ngram_size=5, stop_n_iter_unchanged=30, temperature=temperature, top_p=top_p, top_k=60, repetition_penalty=1.1)
    strategy = AStrategy(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        end_tokens=[tokenizer.eos_token_id] + end_tokens,
        invalid_slices=[],
        repetition_penalty=args.repetition_penalty,
    )

    outputs = filling_sequence(
        model,
        seq,
        batch_size=1,
        get_masks_and_position_ids=get_func,
        strategy=typing.cast(BaseStrategy, strategy),
        pre_image=[pre_image],
        image=torch_image,
    )[0]

    # ---------------
    # port from inference_glm.py, more general than chat mode
    # clip -1s and fill back generated things into seq

    if type(outputs) is not list:
        output_list = outputs.tolist()
    else:
        output_list = outputs

    # only process the first one
    output_ids = output_list[0]
    unfinished = output_ids.index(-1) if -1 in output_ids else len(output_ids)
    if output_ids[unfinished - 1] == tokenizer.eos_token_id:
        unfinished -= 1
    bog = output_ids.index(tokenizer.bos_token_id)

    left_ctx = output_ids[:mask_position]
    generated = output_ids[bog + 1 : unfinished]
    right_ctx = output_ids[mask_position + 1 : bog]
    assert len(right_ctx) == 0

    text = tokenizer.decode(generated[prefix_len:])
    assert strategy.preds is not None and strategy.probs is not None
    # len(preds) == len(generation[prefix_len:]) or len(preds) == len(generation[prefix_len:]) + 1 (eos)
    return text, strategy.preds[0], strategy.probs[0]


def decode_mmmd(model, tokenizer, example, yes_token, no_token, eol_token, args):
    context = example["prompt"]

    def get_image(path):
        if path is None:
            return None
        else:
            return Image.open(args.image_base_dir + "/" + path).convert("RGB")

    predicted = {}

    # fields & response
    prefix = ""
    for field in ("当前目标", "当前实体", "剩余回合", "回复图片", "\n### 答"):
        prefix += field + ": "
        text, tokens, probs = mmmd_generate(
            model,
            tokenizer,
            context,
            get_image(example["input_image"]),
            prefix,
            [eol_token],
            args,
        )

        predicted[field] = text
        if args.error_propagation:
            prefix += predicted[field] + "\n"
        else:
            prefix += example[field] + "\n"

    # image selection with ground truth structure
    prefix = ""
    for field in ("当前目标", "当前实体", "剩余回合", "回复图片"):
        prefix += field + ": " + example[field] + "\n"
    prefix += "图片合适: "

    if "pos_image" in example:
        confidence = []
        for img in [example["pos_image"]] + example["neg_images"]:
            text, tokens, probs = mmmd_generate(
                model,
                tokenizer,
                context,
                get_image(img),
                prefix,
                [eol_token],
                args,
            )
            yes, no = 0.0, 0.0
            for prob in probs:
                yes += prob[yes_token].item()
                no += prob[no_token].item()
            confidence.append((yes, no))

        predicted["pos_image"] = confidence[0]
        predicted["neg_images"] = confidence[1:]

    return predicted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_length", type=int, default=2048, help="max length of the total sequence"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.4, help="top p for nucleus sampling"
    )
    parser.add_argument(
        "--top_k", type=int, default=100, help="top k for top k sampling"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="temperature for sampling"
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.2, help="repetition penalty"
    )
    parser.add_argument(
        "--from_pretrained", type=str, default="visualglm-6b", help="pretrained ckpt"
    )
    parser.add_argument("--error_propagation", action="store_true")
    parser.add_argument("--test_data", type=str, help="test file path")
    parser.add_argument("--image_base_dir", type=str, default=".")
    parser.add_argument("--result_file", type=str)

    args = parser.parse_args()
    args.from_pretrained = "/home/bchen/projects/mmmd/VisualGLM-6B/checkpoints/finetune-visualglm-6b-01-27-00-25/"
    args.image_base_dir = "/home/bchen/projects/mmmd/data/imgs/"
    args.test_data = "/home/bchen/projects/mmmd/data/data/test.jsonl"
    args.result_file = "./prediction_error_prop.jsonl"
    args.error_propagation = True

    examples = load_mmmd_data(args)

    tokenizer = AutoTokenizer.from_pretrained(
        "THUDM/chatglm-6b", trust_remote_code=True
    )
    model, tokenizer = get_mmmd_model(args)

    # what is this special token 5
    yes_token = tokenizer.encode("是", add_special_tokens=False)[1]
    no_token = tokenizer.encode("否", add_special_tokens=False)[1]
    eol_token = tokenizer.encode("\n", add_special_tokens=False)[1]

    with open(args.result_file, "w") as f:
        for example in tqdm(examples):
            pred = decode_mmmd(
                model, tokenizer, example, yes_token, no_token, eol_token, args
            )
            example["prediction"] = pred
            print(json.dumps(example, ensure_ascii=False), file=f, flush=True)


if __name__ == "__main__":
    main()
