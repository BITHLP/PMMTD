import json
from copy import deepcopy

import torch
from sat.model.base_model import BaseMixin
from sat.model.official import ChatGLMModel
from sat.resources.urls import MODEL_URLS

from .blip2 import BLIP2

MODEL_URLS["visualglm-6b"] = "r2://visualglm-6b.zip"


class ImageMixin(BaseMixin):
    def __init__(self, args):
        super().__init__()
        self.args = deepcopy(args)
        if hasattr(args, "model_parallel_size"):
            args.eva_args["model_parallel_size"] = args.model_parallel_size
            args.qformer_args["model_parallel_size"] = args.model_parallel_size
        self.model = BLIP2(args.eva_args, args.qformer_args)

    def word_embedding_forward_each(self, input_ids, image_emb, pre_image):
        if image_emb is None or pre_image > input_ids.shape[0]:
            return self.transformer.word_embeddings(input_ids)

        # the image is inserted after 问：<img>, override 32 pads
        pre_id, pads, post_id = torch.tensor_split(
            input_ids, [pre_image, pre_image + self.args.image_length]
        )
        pre_txt_emb = self.transformer.word_embeddings(pre_id)
        post_txt_emb = self.transformer.word_embeddings(post_id)
        return torch.cat([pre_txt_emb, image_emb, post_txt_emb])

    def word_embedding_forward(self, input_ids, output_cross_layer, **kw_args):
        if kw_args.get("image", None) is None:
            return self.transformer.word_embeddings(input_ids)
        image_embs = self.model(image=kw_args["image"])

        batch = []
        for ids, image_emb, pre_image in zip(
            input_ids, image_embs, kw_args["pre_image"]
        ):
            batch.append(self.word_embedding_forward_each(ids, image_emb, pre_image))

        batch_ids = torch.stack(batch)
        return batch_ids


class VisualGLMModel(ChatGLMModel):
    def __init__(self, args, transformer=None, **kwargs):
        super().__init__(args, transformer=transformer, **kwargs)
        self.image_length = args.image_length
        self.add_mixin("eva", ImageMixin(args))

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group("VisualGLM", "VisualGLM Configurations")
        group.add_argument("--image_length", type=int, default=32)
        group.add_argument("--eva_args", type=json.loads, default={})
        group.add_argument("--qformer_args", type=json.loads, default={})
        return super().add_model_specific_args(parser)
