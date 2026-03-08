from typing import cast

import torch
import torch.nn as nn
import pandas as pd
from dotmap import DotMap
from einops import rearrange
from .clip import clip
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .selector_model import SelectorModel
from .anomaly_clip import AnomalyCLIP
from .text_encoder import TextEncoder
from .temporal_model import TemporalModel

_tokenizer = _Tokenizer()


class MetaNet(nn.Module):
    def __init__(self, vis_dim, ctx_dim):
        super().__init__()
        self.linear1 = nn.Linear(vis_dim, vis_dim // 16)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(vis_dim // 16, ctx_dim)

        nn.init.zeros_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


class PromptLearnerCoOp(nn.Module):
    def __init__(self, config, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = config.n_ctx
        ctx_init = config.ctx_init
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)
        self.meta_net = MetaNet(vis_dim, ctx_dim).to(dtype)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = "end"

    def forward(self, im_features):
        prefix = cast(torch.Tensor, self.token_prefix)
        suffix = cast(torch.Tensor, self.token_suffix)
        ctx = self.ctx
        bias = self.meta_net(im_features)
        bias = bias.unsqueeze(1)
        ctx = ctx.unsqueeze(0)
        ctx_shifted = ctx + bias

        ctx_shifted = ctx_shifted.unsqueeze(1)
        prefix = prefix.unsqueeze(0)
        suffix = suffix.unsqueeze(0)

        batch_size = ctx_shifted.shape[0]

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix.expand(batch_size, -1, -1, -1),
                    ctx_shifted.expand(-1, self.n_cls, -1, -1),
                    suffix.expand(batch_size, -1, -1, -1),
                ],
                dim=2,
            )
        else:
            raise NotImplementedError(
                f"class_token_position {self.class_token_position} not implemented for PromptLearnerCoOp"
            )

        return prompts


class SelectorModelDynamic(SelectorModel):
    def forward(
        self,
        image_features,
        text_features,
        labels,
        ncentroid,
        test_mode,
    ):
        B, T, D = image_features.shape
        C = text_features.shape[1]

        text_features_except_normal = torch.cat(
            (
                text_features[:, : self.normal_id],
                text_features[:, (self.normal_id + 1) :],
            ),
            dim=1,
        )

        if self.recentering_mode == "legacy":
            text_features = text_features_except_normal - ncentroid
            image_features = image_features - ncentroid
        else:
            text_features = text_features_except_normal

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = torch.einsum("btd,bcd->btc", image_features, text_features)

        logits = logits.view(-1, C - 1)
        logits = self.bn_layer(logits)

        if test_mode:
            return logits
        else:
            logits = logits.view(B, T, C - 1)

            topk_mask, bottomk_mask = self.generate_mask(logits)
            topk_mask = topk_mask.to(image_features.device)
            bottomk_mask = bottomk_mask.to(image_features.device)

            logits_topk, idx_topk = self.select_topk(logits, labels, topk_mask)
            idx_topk_abn, idx_topk_nor = (
                idx_topk[: idx_topk.shape[0] // 2],
                idx_topk[idx_topk.shape[0] // 2 :],
            )

            logits_bottomk, idx_bottomk = self.select_bottomk(logits, labels, bottomk_mask)
            idx_bottomk_abn = idx_bottomk[: idx_bottomk.shape[0] // 2]

            logits = logits.view(-1, logits.shape[-1])
            logits_topk = logits_topk.view(-1, logits_topk.shape[-1])
            logits_bottomk = logits_bottomk.view(-1, logits_bottomk.shape[-1])

            return (
                logits,
                logits_topk,
                logits_bottomk,
                idx_topk_abn,
                idx_topk_nor,
                idx_bottomk_abn,
                text_features,
            )


class AnomalyCLIPCoOp(AnomalyCLIP):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        config = DotMap(**kwargs)

        clip_model, _ = clip.load(self.arch, device="cpu")
        clip_model.float()

        classes_df = pd.read_csv(self.labels_file)
        classnames = sorted(c for i, c in classes_df.values.tolist())

        self.prompt_learner = PromptLearnerCoOp(config, classnames, clip_model)
        self.selector_model = SelectorModelDynamic(
            classnames=classnames,
            normal_id=self.normal_id,
            logit_scale=clip_model.logit_scale,
            num_segments=self.num_segments,
            seg_length=self.seg_length,
            select_idx_dropout_topk=self.select_idx_dropout_topk,
            select_idx_dropout_bottomk=self.select_idx_dropout_bottomk,
            num_topk=self.num_topk,
            num_bottomk=self.num_bottomk,
            recentering_mode=self.recentering_mode,
        )

    def get_text_features(self, video_features=None):
        if video_features is None:
            raise ValueError("video_features is required for AnomalyCLIPCoOp.get_text_features")

        prompts = self.prompt_learner(video_features)
        B, C, L, D = prompts.shape

        prompts = prompts.view(B * C, L, D)
        tokenized_prompts = self.tokenized_prompts.repeat(B, 1).to(prompts.device)

        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features.view(B, C, -1)
        return text_features

    def forward(
        self,
        image_features,
        labels,
        ncentroid,
        segment_size=1,
        test_mode=False,
    ):
        ncentroid = ncentroid.to(image_features.device)

        if test_mode:
            if not self.load_from_features:
                b, t, c, h, w = image_features.size()
                image_features = image_features.view(-1, c, h, w)
                image_features = self.image_encoder(
                    image_features
                )
                image_features = rearrange(
                    image_features,
                    "(b ncrops n s l) d -> b ncrops (n s l) d",
                    ncrops=self.ncrops,
                    n=self.num_segments,
                    s=segment_size,
                    l=self.seg_length,
                )
            b, ncrops, t, d = image_features.shape

            image_features = image_features.view(-1, t, d)

            video_features = image_features.mean(dim=1)
            text_features = self.get_text_features(video_features)

            similarity = self.selector_model(
                image_features, text_features, labels, ncentroid, test_mode
            )

            if self.recentering_mode == "legacy" or self.recentering_mode == "image_only":
                image_features = image_features - ncentroid

            features = self.get_temporal_model_input(image_features, similarity)

            scores = self.temporal_model(features, segment_size, test_mode)

            similarity = similarity.repeat_interleave(self.stride, dim=0)
            scores = scores.repeat_interleave(self.stride, dim=0)

            scores = scores.view(-1)

            return similarity, scores

        else:
            if not self.load_from_features:
                (b, t, c, h, w) = image_features.size()
                image_features = image_features.view(-1, c, h, w)
                image_features = self.image_encoder(image_features)
                image_features = rearrange(
                    image_features,
                    "(b ncrops n l) d -> b ncrops (n l) d",
                    ncrops=self.ncrops,
                    n=self.num_segments,
                    l=self.seg_length,
                )

            (b, ncrops, t, d) = image_features.shape
            image_features = image_features.view(b * ncrops, t, d)

            video_features = image_features.mean(dim=1)
            text_features = self.get_text_features(video_features)

            (
                logits,
                logits_topk,
                logits_bottomk,
                idx_topk_abn,
                idx_topk_nor,
                idx_bottomk_abn,
                text_directions,
            ) = self.selector_model(
                image_features,
                text_features,
                labels,
                ncentroid,
                test_mode,
            )

            if self.recentering_mode == "legacy" or self.recentering_mode == "image_only":
                image_features = image_features - ncentroid

            features = self.get_temporal_model_input(image_features, logits)

            scores = self.temporal_model(features, segment_size, test_mode)
            scores = scores.view(-1)

            return (
                logits,
                logits_topk,
                scores,
                idx_topk_abn,
                idx_topk_nor,
                idx_bottomk_abn,
                text_directions,
            )
