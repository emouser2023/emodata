# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

import torch.nn.functional as F
import torch.nn as nn



import torch
import torch.nn as nn
import torch.nn.functional as F

# class SentimentDivergenceAwareTriModalLoss(nn.Module):
#     def __init__(self, tau=1.0, beta=1.0, lambda_=0.5, delta=0.1):
#         super().__init__()
#         self.tau = tau  # Temperature for contrastive loss
#         self.beta = beta  # Scaling factor for reweighting
#         self.lambda_ = lambda_  # Weight for divergence penalty
#         self.delta = delta  # Divergence term coefficient

#     def compute_dsim(self, v, t, a, st, sa):
#         """Compute divergence-aware similarity."""
#         # Pairwise dot products (shape: [B, B])
#         vt = v @ t.T  # v_i · t_j
#         va = v @ a.T  # v_i · a_j
#         ta = t @ a.T  # t_i · a_j

#         # Sentiment divergence (KL between text and audio logits)
#         pt = F.softmax(st, dim=-1)  # [B, 7]
#         pa = F.softmax(sa, dim=-1)  # [B, 7]
#         d = torch.sum(pt * (pt.log() - pa.log()), dim=-1)  # [B]

#         # Dynamic weights
#         alpha = 1 / (1 + d)  # [B]
#         beta = 1 / (1 + d / 2)  # [B]
#         gamma = torch.ones_like(alpha)  # [B]

#         # Normalize weights to sum to 3
#         weight_sum = alpha + beta + gamma  # [B]
#         alpha = alpha / weight_sum * 3  # [B]
#         beta = beta / weight_sum * 3   # [B]
#         gamma = gamma / weight_sum * 3  # [B]

#         # Expand weights for broadcasting: [B] -> [B, 1]
#         alpha = alpha.unsqueeze(1)  # [B, 1]
#         beta = beta.unsqueeze(1)
#         gamma = gamma.unsqueeze(1)

#         # Compute dsim for all pairs
#         dsim = alpha * vt + beta * va + gamma * ta  # [B, B]
#         return dsim, d

#     def compute_w(self, st, sa, d):
#         """Compute reweighting term w_{i,j}."""
#         B = st.shape[0]
#         pt = F.softmax(st, dim=-1)  # [B, 7]
#         pa = F.softmax(sa, dim=-1)  # [B, 7]

#         # KL divergences
#         kl_t = torch.zeros(B, B, device=st.device)  # [B, B]
#         kl_a = torch.zeros(B, B, device=st.device)
#         for i in range(B):
#             for j in range(B):
#                 kl_t[i, j] = F.kl_div(pt[i].log(), pt[j], reduction='sum', log_target=False)
#                 kl_a[i, j] = F.kl_div(pa[i].log(), pa[j], reduction='sum', log_target=False)

#         # Divergence term: d_i * d_j
#         d_outer = d.unsqueeze(1) * d.unsqueeze(0)  # [B, B]

#         # Total distance
#         dist = kl_t + kl_a + self.delta * d_outer  # [B, B]
#         w = self.beta / (dist + 1e-8)  # Avoid division by zero, [B, B]
#         w.fill_diagonal_(0)  # w_{i,i} = 0
#         return w

#     def forward(self, v, t, a, st, sa):
#         """Compute SDATML v2 loss."""
#         B = v.shape[0]
#         device = v.device

#         # Normalize embeddings (CLIP-style)
#         v = F.normalize(v, dim=-1)
#         t = F.normalize(t, dim=-1)
#         a = F.normalize(a, dim=-1)

#         # Alignment term
#         dsim, d = self.compute_dsim(v, t, a, st, sa)  # [B, B], [B]
#         w = self.compute_w(st, sa, d)  # [B, B]

#         # Logits: dsim / tau - w
#         logits = dsim / self.tau - w  # [B, B]
#         labels = torch.arange(B, device=device)  # [0, 1, ..., B-1]

#         # Contrastive loss (symmetric not needed, single direction suffices for triplet)
#         align_loss = F.cross_entropy(logits, labels)

#         # Divergence penalty
#         vt_dist = torch.sum((v - t) ** 2, dim=-1)  # [B]
#         va_dist = torch.sum((v - a) ** 2, dim=-1)  # [B]
#         ta_dist = torch.sum((t - a) ** 2, dim=-1)  # [B]
#         div_loss = torch.mean(d * (vt_dist + va_dist + ta_dist))

#         # Total loss
#         total_loss = align_loss + self.lambda_ * div_loss
#         return total_loss

# class KLLoss(nn.Module):
#     """Loss that uses a 'hinge' on the lower bound.
#     This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
#     also smaller than that threshold.
#     args:
#         error_matric:  What base loss to use (MSE by default).
#         threshold:  Threshold to use for the hinge.
#         clip:  Clip the loss if it is above this value.
#     """

#     def __init__(self, error_metric=nn.KLDivLoss(size_average=True, reduce=True)):
#         super().__init__()
#         print('=========using KL Loss=and has temperature and * bz==========')
#         self.error_metric = error_metric

#     def forward(self, prediction, label):
#         batch_size = prediction.shape[0]
#         probs1 = F.log_softmax(prediction, 1)
#         probs2 = F.softmax(label * 10, 1)
#         loss = self.error_metric(probs1, probs2) * batch_size
#         return loss
    




import sys
import torch
from torch import distributed as dist, nn as nn
from torch.nn import functional as F
if sys.platform == 'linux':
    import torch.distributed.nn

# def gather_features(
#         video_features,
#         text_features,
#         audio_features,
#         text_sentiment_score,
#         audio_sentiment_score,
#         local_loss=False,
#         gather_with_grad=False,
#         rank=0,
#         world_size=1
# ):
#     all_text_sentiment_score = None
#     all_audio_sentiment_score = None
#     if gather_with_grad and sys.platform == 'linux':
#         all_video_features = torch.cat(torch.distributed.nn.all_gather(video_features), dim=0)
#         all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
#         all_audio_features = torch.cat(torch.distributed.nn.all_gather(audio_features), dim=0)
#         if text_sentiment_score is not None:
#             all_text_sentiment_score = torch.cat(torch.distributed.nn.all_gather(text_sentiment_score), dim=0)
#         if audio_sentiment_score is not None:
#             all_audio_sentiment_score = torch.cat(torch.distributed.nn.all_gather(audio_sentiment_score), dim=0)
#     else:
#         gathered_video_features = [torch.zeros_like(video_features) for _ in range(world_size)]
#         gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
#         gathered_audio_features = [torch.zeros_like(audio_features) for _ in range(world_size)]
#         dist.all_gather(gathered_video_features, video_features)
#         dist.all_gather(gathered_text_features, text_features)
#         dist.all_gather(gathered_audio_features, audio_features)
#         if not local_loss:
#             gathered_video_features[rank] = video_features
#             gathered_text_features[rank] = text_features
#             gathered_audio_features[rank] = audio_features
#         all_video_features = torch.cat(gathered_video_features, dim=0)
#         all_text_features = torch.cat(gathered_text_features, dim=0)
#         all_audio_features = torch.cat(gathered_audio_features, dim=0)
#         if text_sentiment_score is not None:
#             gathered_text_sentiment_score = [torch.zeros_like(text_sentiment_score) for _ in range(world_size)]
#             dist.all_gather(gathered_text_sentiment_score, text_sentiment_score)
#             if not local_loss:
#                 gathered_text_sentiment_score[rank] = text_sentiment_score
#             all_text_sentiment_score = torch.cat(gathered_text_sentiment_score, dim=0)
#         if audio_sentiment_score is not None:
#             gathered_audio_sentiment_score = [torch.zeros_like(audio_sentiment_score) for _ in range(world_size)]
#             dist.all_gather(gathered_audio_sentiment_score, audio_sentiment_score)
#             if not local_loss:
#                 gathered_audio_sentiment_score[rank] = audio_sentiment_score
#             all_audio_sentiment_score = torch.cat(gathered_audio_sentiment_score, dim=0)
#     return all_video_features, all_text_features, all_audio_features, all_text_sentiment_score, all_audio_sentiment_score

# def get_sentiment_weights(text_score, audio_score, all_text_score, all_audio_score):
#     text_prob = F.softmax(text_score, dim=-1)
#     audio_prob = F.softmax(audio_score, dim=-1)
#     all_text_prob = F.softmax(all_text_score, dim=-1)
#     all_audio_prob = F.softmax(all_audio_score, dim=-1)
    
#     # Divergence d_i
#     d_i = F.kl_div(F.log_softmax(text_score, dim=-1), audio_prob, reduction='none').sum(dim=-1, keepdim=True)
    
#     # Alignment weights
#     alpha = 1.0 / (1.0 + d_i)
#     beta = 1.0 / (1.0 + d_i / 2.0)
#     gamma = torch.exp(-d_i)
    
#     # Reweighting terms
#     kl_text = F.kl_div(F.log_softmax(text_score, dim=-1).unsqueeze(1), all_text_prob.unsqueeze(0), reduction='none').sum(dim=-1)
#     kl_audio = F.kl_div(F.log_softmax(audio_score, dim=-1).unsqueeze(1), all_audio_prob.unsqueeze(0), reduction='none').sum(dim=-1)
#     w_t = 1.0 / kl_text.clamp(min=1e-6)
#     w_a = 1.0 / kl_audio.clamp(min=1e-6)
#     w_ta = 1.0 / torch.minimum(kl_text, kl_audio).clamp(min=1e-6)
#     w_t.fill_diagonal_(0)
#     w_a.fill_diagonal_(0)
#     w_ta.fill_diagonal_(0)
    
#     return alpha, beta, gamma, w_t, w_a, w_ta

# class SMTCLoss(nn.Module):
#     def __init__(
#             self,
#             local_loss=False,
#             gather_with_grad=False,
#             cache_labels=False,
#             rank=0,
#             world_size=1,
#             sentiment_scale=1.0,
#             separation_lambda=0.5,
#             regularization_mu=0.5,
#             min_similarity=0.5,
#             separation_tau=1.0
#     ):
#         super().__init__()
#         self.local_loss = local_loss
#         self.gather_with_grad = gather_with_grad
#         self.cache_labels = cache_labels
#         self.rank = rank
#         self.world_size = world_size
#         self.sentiment_scale = sentiment_scale
#         self.separation_lambda = separation_lambda
#         self.regularization_mu = regularization_mu
#         self.min_similarity = min_similarity
#         self.separation_tau = separation_tau

#         self.prev_num_logits = 0
#         self.labels = {}

#     def forward(self, video_features, text_features, audio_features, logit_scale, text_sentiment_score, audio_sentiment_score):
#         device = video_features.device

#         video_features = F.normalize(video_features, dim=-1)
#         text_features = F.normalize(text_features, dim=-1)
#         audio_features = F.normalize(audio_features, dim=-1)
        
#         if self.world_size > 1:
#             all_video_features, all_text_features, all_audio_features, all_text_sentiment_score, all_audio_sentiment_score = gather_features(
#                 video_features, text_features, audio_features, text_sentiment_score, audio_sentiment_score,
#                 self.local_loss, self.gather_with_grad, self.rank, self.world_size
#             )
#             if self.local_loss:
#                 v_features = video_features
#                 t_features = text_features
#                 a_features = audio_features
#                 t_sentiment = text_sentiment_score
#                 a_sentiment = audio_sentiment_score
#             else:
#                 v_features = all_video_features
#                 t_features = all_text_features
#                 a_features = all_audio_features
#                 t_sentiment = all_text_sentiment_score
#                 a_sentiment = all_audio_sentiment_score
#         else:
#             v_features = video_features
#             t_features = text_features
#             a_features = audio_features
#             t_sentiment = text_sentiment_score
#             a_sentiment = audio_sentiment_score

#         # Pairwise similarities
#         vt_logits = logit_scale * v_features @ t_features.T
#         va_logits = logit_scale * v_features @ a_features.T
#         ta_logits = logit_scale * t_features @ a_features.T

#         # Sentiment-guided weights
#         alpha, beta, gamma, w_t, w_a, w_ta = get_sentiment_weights(t_sentiment, a_sentiment, t_sentiment, a_sentiment)

#         # Staged alignment logits
#         vt_logits_weighted = alpha * vt_logits
#         va_logits_weighted = beta * va_logits
#         ta_logits_weighted = gamma * ta_logits

#         # Apply reweighting
#         vt_logits_adjusted = vt_logits_weighted - self.sentiment_scale * w_t
#         va_logits_adjusted = va_logits_weighted - self.sentiment_scale * w_a
#         ta_logits_adjusted = ta_logits_weighted - self.sentiment_scale * w_ta

#         # Labels
#         num_logits = vt_logits.shape[0]
#         if self.prev_num_logits != num_logits or device not in self.labels:
#             labels = torch.arange(num_logits, device=device, dtype=torch.long)
#             if self.world_size > 1 and self.local_loss:
#                 labels = labels + num_logits * self.rank
#             if self.cache_labels:
#                 self.labels[device] = labels
#                 self.prev_num_logits = num_logits
#         else:
#             labels = self.labels[device]

#         # Staged alignment loss
#         staged_loss = (
#             F.cross_entropy(vt_logits_adjusted, labels) +
#             F.cross_entropy(va_logits_adjusted, labels) +
#             F.cross_entropy(ta_logits_adjusted, labels)
#         )

#         # Sentiment-modulated separation
#         kl_text = F.kl_div(F.log_softmax(t_sentiment, dim=-1).unsqueeze(1), F.softmax(t_sentiment, dim=-1).unsqueeze(0), reduction='none').sum(dim=-1)
#         kl_audio = F.kl_div(F.log_softmax(a_sentiment, dim=-1).unsqueeze(1), F.softmax(a_sentiment, dim=-1).unsqueeze(0), reduction='none').sum(dim=-1)
#         s_ij = torch.maximum(kl_text, kl_audio)
#         mask = torch.ones_like(s_ij, dtype=torch.bool)
#         mask.fill_diagonal_(0)
#         distances = (v_features.unsqueeze(1) - v_features.unsqueeze(0)).pow(2).sum(dim=-1) + \
#                     (t_features.unsqueeze(1) - t_features.unsqueeze(0)).pow(2).sum(dim=-1) + \
#                     (a_features.unsqueeze(1) - a_features.unsqueeze(0)).pow(2).sum(dim=-1)
#         separation_loss = - (s_ij * torch.exp(-distances / self.separation_tau) * mask).sum() / mask.sum()

#         # Adaptive divergence regularization
#         d_i = F.kl_div(F.log_softmax(t_sentiment, dim=-1), F.softmax(a_sentiment, dim=-1), reduction='none').sum(dim=-1)
#         reg_vt = F.relu(self.min_similarity - vt_logits.diagonal()).mean()
#         reg_va = F.relu(self.min_similarity - va_logits.diagonal()).mean()
#         reg_ta = F.relu(self.min_similarity - ta_logits.diagonal()).mean()
#         regularization_loss = (d_i * (reg_vt + reg_va + reg_ta)).mean()

#         # Total loss
#         total_loss = staged_loss + self.separation_lambda * separation_loss + self.regularization_mu * regularization_loss
#         # total_loss = staged_loss #+ self.separation_lambda * separation_loss + self.regularization_mu * regularization_loss
#         return total_loss
    

def gather_features(
        video_features,
        text_features,
        audio_features,
        text_sentiment_score,
        audio_sentiment_score,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1
):
    all_text_sentiment_score = None
    all_audio_sentiment_score = None
    if gather_with_grad and sys.platform == 'linux':
        all_video_features = torch.cat(torch.distributed.nn.all_gather(video_features), dim=0)
        all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        all_audio_features = torch.cat(torch.distributed.nn.all_gather(audio_features), dim=0)
        if text_sentiment_score is not None:
            all_text_sentiment_score = torch.cat(torch.distributed.nn.all_gather(text_sentiment_score), dim=0)
        if audio_sentiment_score is not None:
            all_audio_sentiment_score = torch.cat(torch.distributed.nn.all_gather(audio_sentiment_score), dim=0)
    else:
        gathered_video_features = [torch.zeros_like(video_features) for _ in range(world_size)]
        gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
        gathered_audio_features = [torch.zeros_like(audio_features) for _ in range(world_size)]
        dist.all_gather(gathered_video_features, video_features)
        dist.all_gather(gathered_text_features, text_features)
        dist.all_gather(gathered_audio_features, audio_features)
        if not local_loss:
            gathered_video_features[rank] = video_features
            gathered_text_features[rank] = text_features
            gathered_audio_features[rank] = audio_features
        all_video_features = torch.cat(gathered_video_features, dim=0)
        all_text_features = torch.cat(gathered_text_features, dim=0)
        all_audio_features = torch.cat(gathered_audio_features, dim=0)
        if text_sentiment_score is not None:
            gathered_text_sentiment_score = [torch.zeros_like(text_sentiment_score) for _ in range(world_size)]
            dist.all_gather(gathered_text_sentiment_score, text_sentiment_score)
            if not local_loss:
                gathered_text_sentiment_score[rank] = text_sentiment_score
            all_text_sentiment_score = torch.cat(gathered_text_sentiment_score, dim=0)
        if audio_sentiment_score is not None:
            gathered_audio_sentiment_score = [torch.zeros_like(audio_sentiment_score) for _ in range(world_size)]
            dist.all_gather(gathered_audio_sentiment_score, audio_sentiment_score)
            if not local_loss:
                gathered_audio_sentiment_score[rank] = audio_sentiment_score
            all_audio_sentiment_score = torch.cat(gathered_audio_sentiment_score, dim=0)
    return all_video_features, all_text_features, all_audio_features, all_text_sentiment_score, all_audio_sentiment_score

def get_sentiment_weight(input_score, target_score):
    input_log_prob = F.log_softmax(input_score, dim=-1)
    target_log_score = F.log_softmax(target_score, dim=-1)
    target_prob = target_log_score.exp()
    sentiment_weight = 1.0 / (F.kl_div(input_log_prob, target_prob, reduction='none').sum(dim=-1).abs().unsqueeze(1) + 1e-6)
    sentiment_weight.fill_diagonal_(0)
    return sentiment_weight

class ExtendedReweightedClipLoss(nn.Module):
    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            sentiment_scale=1.0
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.sentiment_scale = sentiment_scale

        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, video_features, text_features, audio_features, logit_scale, text_sentiment_score, audio_sentiment_score):
        device = video_features.device

        video_features = F.normalize(video_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        audio_features = F.normalize(audio_features, dim=-1)
        
        if self.world_size > 1:
            all_video_features, all_text_features, all_audio_features, all_text_sentiment_score, all_audio_sentiment_score = gather_features(
                video_features, text_features, audio_features, text_sentiment_score, audio_sentiment_score,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size
            )
            if self.local_loss:
                v_features = video_features
                t_features = text_features
                a_features = audio_features
                t_sentiment = text_sentiment_score
                a_sentiment = audio_sentiment_score
            else:
                v_features = all_video_features
                t_features = all_text_features
                a_features = all_audio_features
                t_sentiment = all_text_sentiment_score
                a_sentiment = all_audio_sentiment_score
        else:
            v_features = video_features
            t_features = text_features
            a_features = audio_features
            t_sentiment = text_sentiment_score
            a_sentiment = audio_sentiment_score

        # Pairwise logits
        logits_per_vt = logit_scale * v_features @ t_features.T
        logits_per_tv = logit_scale * t_features @ v_features.T
        logits_per_va = logit_scale * v_features @ a_features.T
        logits_per_av = logit_scale * a_features @ v_features.T
        logits_per_ta = logit_scale * t_features @ a_features.T
        logits_per_at = logit_scale * a_features @ t_features.T

        # Sentiment weights
        vt_weight = get_sentiment_weight(t_sentiment, t_sentiment)  # Text-guided for video-text
        va_weight = get_sentiment_weight(a_sentiment, a_sentiment)  # Audio-guided for video-audio
        ta_weight = get_sentiment_weight(t_sentiment, a_sentiment)  # Text-audio mix

        # Apply reweighting
        logits_per_vt_adjusted = logits_per_vt - self.sentiment_scale * vt_weight
        logits_per_tv_adjusted = logits_per_tv - self.sentiment_scale * vt_weight
        logits_per_va_adjusted = logits_per_va - self.sentiment_scale * va_weight
        logits_per_av_adjusted = logits_per_av - self.sentiment_scale * va_weight
        logits_per_ta_adjusted = logits_per_ta - self.sentiment_scale * ta_weight
        logits_per_at_adjusted = logits_per_at - self.sentiment_scale * ta_weight

        # Labels
        num_logits = v_features.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        # Total loss: average of three pairwise contrastive losses
        total_loss = (
            F.cross_entropy(logits_per_vt_adjusted, labels) +
            F.cross_entropy(logits_per_tv_adjusted, labels) +
            F.cross_entropy(logits_per_va_adjusted, labels) +
            F.cross_entropy(logits_per_av_adjusted, labels) +
            F.cross_entropy(logits_per_ta_adjusted, labels) +
            F.cross_entropy(logits_per_at_adjusted, labels)
        ) / 6  # Average over three modality pairs

        return total_loss

# Test with random tensors
if __name__ == "__main__":
    # Dummy data
    B, D, S = 2, 512, 7  # Batch size, feature dim, sentiment classes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    video_features = torch.randn(B, D, device=device)
    text_features = torch.randn(B, D, device=device)
    audio_features = torch.randn(B, D, device=device)
    text_sentiment_score = torch.tensor([
        [0.1, 0.05, 0.1, 0.05, 0.6, 0.05, 0.05],
        [0.05, 0.05, 0.05, 0.7, 0.05, 0.05, 0.05]
    ], device=device, dtype=torch.float32)
    audio_sentiment_score = torch.tensor([
        [0.05, 0.05, 0.05, 0.7, 0.05, 0.05, 0.05],
        [0.1, 0.05, 0.1, 0.05, 0.6, 0.05, 0.05]
    ], device=device, dtype=torch.float32)
    logit_scale = torch.tensor(1.0, device=device)  # 1/tau

    # Normalize features (CLIP-style)
    video_features = F.normalize(video_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    audio_features = F.normalize(audio_features, dim=-1)

    # Initialize loss
    loss_fn = ExtendedReweightedClipLoss(world_size=1)  # Single GPU
    loss = loss_fn(video_features, text_features, audio_features, logit_scale, text_sentiment_score, audio_sentiment_score)
    
    print(f"Video features shape: {video_features.shape}")
    print(f"Text features shape: {text_features.shape}")
    print(f"Audio features shape: {audio_features.shape}")
    print(f"Text sentiment shape: {text_sentiment_score.shape}")
    print(f"Audio sentiment shape: {audio_sentiment_score.shape}")
    print(f"Loss: {loss.item()}")