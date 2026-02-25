import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet50
from torchvision.models.densenet import densenet121

from timm.models import create_model

import numpy as np
from resnet3d import ResNet3DEncoder
import timm
from util import task_importance_weights,SimpleFusion,RankEmbed,RankReference,GeMPool,TopKAttnPool,FiLMBlock,StepQueryFusion
from diffloss import DiffLoss
def prepare_model_resnet3d(arch='vit_base_patch16_224', pretrained=True):
    # 使用 timm 加载预训练的 ViT-B 模型
    # model = timm.create_model(arch, pretrained=pretrained)
    # print('model',model)
    model = ResNet3DEncoder(backbone='r3d_18', pretrained=True)

    # 打印模型信息
    # print('-------------------------------vit-------------------------', model)

    return model

def prepare_model_vit(arch='vit_base_patch16_224', pretrained=True):
    # 使用 timm 加载预训练的 ViT-B 模型
    model = timm.create_model(arch, pretrained=pretrained)
    # print('model',model)

    # 打印模型信息
    # print('-------------------------------vit-------------------------', model)

    return model
class Resnet3dFeatureExtractor(nn.Module):
    def __init__(self, vit_model,
                 decoder_embed_dim=512,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 diffloss_d=3,
                 diffloss_w=1024,
                 num_sampling_steps='ddim25',
                 diffusion_batch_mul=4,
                 grad_checkpointing=False,
                 ):
        super(Resnet3dFeatureExtractor, self).__init__()
        self.vit_model = vit_model
        # self.head = self.vit_model.head
        self.token_embed_dim = 1
        self.diffloss = DiffLoss(
            target_channels=self.token_embed_dim,
            z_channels=decoder_embed_dim,
            width=diffloss_w,
            depth=diffloss_d,
            num_sampling_steps=num_sampling_steps,
            grad_checkpointing=grad_checkpointing
        )
        self.diffusion_batch_mul = diffusion_batch_mul
        self.SimpleFusion = SimpleFusion()
        self.rank = RankReference()
        self.lambda_o = 0.2
        self.lambda_m = 0.2
        # self.pool = TopKAttnPool(k=round(0.1 * 196))   # 10% patch
        self.pool = GeMPool(p_init=2.5)
        self.query_builder = StepQueryFusion(d_feat=512, d_model=512, num_steps=4)
    def forward(self, image,p,target,t,real_target, return_debug: bool = False):
        target = target.unsqueeze(1)
        target = target.repeat(self.diffusion_batch_mul, 1)
        debug_info = {}
        if t == 0:
            x = self.vit_model(image.float())
            cls = x
            rank_feature = cls
            rank_feature = rank_feature.repeat(self.diffusion_batch_mul, 1)
            z = cls.repeat(self.diffusion_batch_mul, 1)
            # z_detached = z.detach()
            loss, probas = self.diffloss(z=z, target=target)

            real_target = real_target.repeat(self.diffusion_batch_mul)
            if return_debug:
                L_ord, L_met, rank_detail = self.rank.order_metric_gol(
                    rank_feature, real_target, return_detail=True
                )
                debug_info["rank_feature"] = rank_feature.detach()
                debug_info["rank_labels"] = real_target.detach()
                debug_info["rank_refs"] = self.rank.R.detach()
                debug_info["loss_ord"] = L_ord.detach()
                debug_info["loss_med"] = L_met.detach()
                debug_info["loss_ord_per_sample"] = rank_detail["ord_each"].detach()
                debug_info["loss_med_per_sample"] = rank_detail["met_each"].detach()
                debug_info["ord_logit_back"] = rank_detail["logit_back"].detach()
                debug_info["ord_logit_fwd"] = rank_detail["logit_fwd"].detach()
                debug_info["ord_target"] = rank_detail["ord_target"].detach()
                debug_info["ord_gap"] = rank_detail["ord_gap"].detach()
                debug_info["med_dist_pos"] = rank_detail["dist_pos"].detach()
                debug_info["med_dist_neg"] = rank_detail["dist_neg"].detach()
                debug_info["med_margin_violation"] = rank_detail["margin_violation"].detach()
                debug_info["med_margin_scalar"] = float(rank_detail["margin_scalar"])
            else:
                L_ord, L_met = self.rank.order_metric_gol(rank_feature, real_target)
            loss = loss + self.lambda_o *L_ord + self.lambda_m * L_met
            fusion_feature = z
            debug_info["step0_fusion_feature"] = fusion_feature.detach()

        if t !=0:
            p = torch.round(p)
            p = torch.clamp(p, 0, 1)  # 限制在 0 和 1 之间
            if return_debug:
                fusion_feature, film_detail = self.query_builder(
                    image, p, t, return_intermediate=True
                )
                debug_info["film_before"] = film_detail["film_before"].detach()
                debug_info["film_after"] = film_detail["film_after"].detach()
                debug_info["film_p_prev"] = film_detail["p_prev"].detach()
            else:
                fusion_feature = self.query_builder(image,p,t)
            

            loss, probas = self.diffloss(z=fusion_feature, target=target)
            real_target = real_target.repeat(self.diffusion_batch_mul)
        if return_debug:
            debug_info["step"] = int(t)
            return loss, probas, fusion_feature, debug_info
        return loss, probas,fusion_feature
    def sample(self, image, t, temperature=1.0, cfg=1.0, sampler="ddim", eta=0.0):
        if t == 0:
            x = self.vit_model(image.float())
        else:
            x = image
        sampled_token_latent = self.diffloss.sample(
            x,
            temperature=temperature,
            cfg=cfg,
            sampler=sampler,
            eta=eta,
        )
        return sampled_token_latent,x
class ViTFeatureExtractor(nn.Module):
    def __init__(self, vit_model,
                 decoder_embed_dim=768,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 diffloss_d=3,
                 diffloss_w=1024,
                 num_sampling_steps='ddim25',
                 diffusion_batch_mul=4,
                 grad_checkpointing=False,
                 ):
        super(ViTFeatureExtractor, self).__init__()
        self.vit_model = vit_model
        self.head = self.vit_model.head
        self.token_embed_dim = 1
        self.diffloss = DiffLoss(
            target_channels=self.token_embed_dim,
            z_channels=decoder_embed_dim,
            width=diffloss_w,
            depth=diffloss_d,
            num_sampling_steps=num_sampling_steps,
            grad_checkpointing=grad_checkpointing
        )
        self.diffusion_batch_mul = diffusion_batch_mul
        self.SimpleFusion = SimpleFusion()
        self.rank = RankReference(num_ranks=5,d_model=1536)
        self.lambda_o = 0.2
        self.lambda_m = 0.2
        self.pool = GeMPool(p_init=2.5)
        self.query_builder = StepQueryFusion(d_feat=768, d_model=768, num_steps=4)
    def forward(self, image,p,target,t,real_target, return_debug: bool = False):
        target = target.unsqueeze(1)
        target = target.repeat(self.diffusion_batch_mul, 1)
        debug_info = {}
        if t == 0:
            x = self.vit_model.forward_features(image.float())
            cls = x[:,0,:]
            gap = x[:,1:,:].mean(1)
            # 这个是vit的
            rank_feature = torch.cat([cls, gap], dim=1)
            rank_feature = rank_feature.repeat(self.diffusion_batch_mul, 1)
            z = cls.repeat(self.diffusion_batch_mul, 1)
            loss, probas = self.diffloss(z=z, target=target)

            real_target = real_target.repeat(self.diffusion_batch_mul)
            if return_debug:
                L_ord, L_met, rank_detail = self.rank.order_metric_gol(
                    rank_feature, real_target, return_detail=True
                )
                debug_info["rank_feature"] = rank_feature.detach()
                debug_info["rank_labels"] = real_target.detach()
                debug_info["rank_refs"] = self.rank.R.detach()
                debug_info["step0_fusion_feature"] = fusion_feature.detach()
                debug_info["loss_ord"] = L_ord.detach()
                debug_info["loss_med"] = L_met.detach()
                debug_info["loss_ord_per_sample"] = rank_detail["ord_each"].detach()
                debug_info["loss_med_per_sample"] = rank_detail["met_each"].detach()
                debug_info["ord_logit_back"] = rank_detail["logit_back"].detach()
                debug_info["ord_logit_fwd"] = rank_detail["logit_fwd"].detach()
                debug_info["ord_target"] = rank_detail["ord_target"].detach()
                debug_info["ord_gap"] = rank_detail["ord_gap"].detach()
                debug_info["med_dist_pos"] = rank_detail["dist_pos"].detach()
                debug_info["med_dist_neg"] = rank_detail["dist_neg"].detach()
                debug_info["med_margin_violation"] = rank_detail["margin_violation"].detach()
                debug_info["med_margin_scalar"] = float(rank_detail["margin_scalar"])
            else:
                L_ord, L_met = self.rank.order_metric_gol(rank_feature, real_target)
            # loss = loss
            loss = loss + self.lambda_o *L_ord + self.lambda_m * L_met
            fusion_feature = z
        if t !=0:
            p = torch.round(p)
            p = torch.clamp(p, 0, 1)  # 限制在 0 和 1 之间
            if return_debug:
                fusion_feature, film_detail = self.query_builder(
                    image, p, t, return_intermediate=True
                )
                debug_info["film_before"] = film_detail["film_before"].detach()
                debug_info["film_after"] = film_detail["film_after"].detach()
                debug_info["film_p_prev"] = film_detail["p_prev"].detach()
            else:
                fusion_feature = self.query_builder(image,p,t)
            
            loss, probas = self.diffloss(z=fusion_feature, target=target)
            real_target = real_target.repeat(self.diffusion_batch_mul)
        if return_debug:
            debug_info["step"] = int(t)
            return loss, probas, fusion_feature, debug_info
        return loss, probas,fusion_feature
    def sample(self, image, t, temperature=1.0, cfg=1.0, sampler="ddim", eta=0.0):
        if t == 0:
            x = self.vit_model.forward_features(image.float())
            x = x[:,0,:]
        else:
            x = image
        sampled_token_latent = self.diffloss.sample(
            x,
            temperature=temperature,
            cfg=cfg,
            sampler=sampler,
            eta=eta,
        )
        return sampled_token_latent,x
