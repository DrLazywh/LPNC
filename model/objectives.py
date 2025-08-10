import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_sdm(image_features, text_features, pid, logit_scale, epsilon=1e-8):
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)
    scores = text_norm @ image_norm.t()
    """
    Similarity Distribution Matching
    """
    batch_size = scores.shape[0]
    pid = pid.reshape((batch_size, 1))  # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    t2i_cosine_theta = scores
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.sum(i2t_loss, dim=1) + torch.sum(t2i_loss, dim=1)

    return loss.sum()/batch_size





def compute_InfoNCE(image_features, text_features, logit_scale):
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)
    scores = text_norm @ image_norm.t()

    # cosine similarity as logits
    logits_per_image = logit_scale * scores
    logits_per_text = logits_per_image.t()

    p1 = F.softmax(logits_per_image, dim=1)
    p2 = F.softmax(logits_per_text, dim=1)

    loss = (- p1.diag().log() - p2.diag().log()) / 2
    return loss


def compute_TAL(image_features, text_features, pid, tau=0.015, margin=0.1):
    # # normalized features
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)
    scores = text_norm @ image_norm.t()

    batch_size = scores.shape[0]
    pid = pid.reshape((batch_size, 1))  # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float().cuda()
    mask = 1 - labels

    alpha_i2t = ((scores / tau).exp() * labels / ((scores / tau).exp() * labels).sum(dim=1, keepdim=True)).detach()
    alpha_t2i = ((scores.t() / tau).exp() * labels / ((scores.t() / tau).exp() * labels).sum(dim=1,
                                                                                             keepdim=True)).detach()

    loss = (-  (alpha_i2t * scores).sum(1) + tau * ((scores / tau).exp() * mask).sum(1).clamp(max=10e35).log() + margin).clamp(min=0) \
           + (-  (alpha_t2i * scores.t()).sum(1) + tau * ((scores.t() / tau).exp() * mask).sum(1).clamp(max=10e35).log() + margin).clamp(min=0)

    return loss.sum()


def compute_TRL(image_features, text_features, pid, tau=0.015, margin=0.1):
    # # normalized features
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)
    scores = text_norm @ image_norm.t()
    batch_size = scores.shape[0]
    pid = pid.reshape((batch_size, 1))  # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float().cuda()
    mask = 1 - labels

    alpha_1 = ((scores / tau).exp() * labels / ((scores / tau).exp() * labels).sum(dim=1, keepdim=True)).detach()
    alpha_2 = ((scores.t() / tau).exp() * labels / ((scores.t() / tau).exp() * labels).sum(dim=1,
                                                                                           keepdim=True)).detach()

    pos_1 = (alpha_1 * scores).sum(1)
    pos_2 = (alpha_2 * scores.t()).sum(1)

    neg_1 = (mask * scores).max(1)[0]
    neg_2 = (mask * scores.t()).max(1)[0]

    cost_1 = (margin + neg_1 - pos_1).clamp(min=0)
    cost_2 = (margin + neg_2 - pos_2).clamp(min=0)
    return (cost_1 + cost_2).sum()



def compute_id(logits, labels):
    """
    Instance loss proposed at http://arxiv.org/abs/1711.05535
    """
    criterion = nn.CrossEntropyLoss(reduction="mean")
    loss = criterion(logits, labels)
    return loss



def compute_mlm(scores, labels):
    ce = nn.CrossEntropyLoss(ignore_index=0)
    return ce(scores, labels)


# --------------------------------------------Cid loss--------------------------------------------
def cosine_similarity_matrix(V, T):
    V_norm = F.normalize(V, p=2, dim=1)
    T_norm = F.normalize(T, p=2, dim=1)
    S = torch.matmul(V_norm, T_norm.t())
    return S
def sample_hard_negatives(S, labels):
    N = S.size(0)
    hard_negatives = {'visual_negatives': [], 'text_negatives': []}
    for i in range(N):
        sorted_idx = torch.argsort(S[i], descending=True)
        for j in sorted_idx:
            if labels[i] != labels[j]: 
                hard_negatives['text_negatives'].append(j.item())
                break
        sorted_idx = torch.argsort(S[:, i], descending=True)
        for j in sorted_idx:
            if labels[i] != labels[j]:
                hard_negatives['visual_negatives'].append(j.item())
                break
    return hard_negatives
def update_labels_for_negatives(labels, hard_negatives, M):
    new_labels = labels.clone()
    N = len(labels)
    for i in range(N):
        new_labels[hard_negatives['text_negatives'][i]] = M + 1
        new_labels[hard_negatives['visual_negatives'][i]] = M + 1
    return new_labels
def create_sample_pairs(V, T, hard_negatives, new_labels, labels):
    N = V.size(0)
    visual_feats = []
    text_feats = []
    all_labels = []
    for i in range(N):
        visual_feats.append(V[i])
        text_feats.append(T[i])
        all_labels.append(labels[i])
        neg_idx = hard_negatives['visual_negatives'][i]
        visual_feats.append(V[neg_idx])
        text_feats.append(T[i])
        all_labels.append(new_labels[neg_idx])
        neg_idx = hard_negatives['text_negatives'][i]
        visual_feats.append(V[i])
        text_feats.append(T[neg_idx])
        all_labels.append(new_labels[neg_idx])
    return torch.stack(visual_feats), torch.stack(text_feats), torch.tensor(all_labels)


def compute_cid(cross_modal_logits1, cross_modal_logits2, labels):
    """
    Cross-modal identity classification loss.
    """
    criterion = nn.CrossEntropyLoss(reduction="mean")
    loss = criterion(cross_modal_logits1, labels) + criterion(cross_modal_logits2, labels)
    return loss / 2
# --------------------------------------------------------------------------------------------------- cid loss