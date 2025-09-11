from prettytable import PrettyTable
import torch
import torch.nn.functional as F
import logging
from knc import kncRetriever, kncRanker

from utils.reranking import re_ranking
from torch.cuda.amp import autocast

def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    if get_mAP:
        similarity = torch.tensor(similarity)  # 将 NumPy 数组转换为 PyTorch 张量
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        # acclerate sort with topk
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )  # q * topk
    pred_labels = g_pids[indices.cpu()]  # q * k
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

    all_cmc = matches[:, :max_rank].cumsum(1) # cumulative sum
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    # all_cmc = all_cmc[topk - 1]

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k

    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices

def get_metrics(similarity, qids, gids, n_, retur_indices=False):
    t2i_cmc, t2i_mAP, t2i_mINP, indices = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
    t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()
    if retur_indices:
        return [n_, t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP, t2i_cmc[0]+ t2i_cmc[4]+ t2i_cmc[9]], indices
    else:
        return [n_, t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP, t2i_cmc[0]+ t2i_cmc[4]+ t2i_cmc[9]]


class Evaluator():
    def __init__(self, img_loader, txt_loader, refer_loader, args):
        self.img_loader = img_loader # gallery
        self.txt_loader = txt_loader # query
        self.refer_loader = refer_loader
        self.logger = logging.getLogger("LPNC.eval")
        self.args = args

    def _compute_embedding(self, model):
        model = model.eval()
        device = next(model.parameters()).device

        qids, gids, qfeats, gfeats, refer_feats, tp,cross = [], [], [], [], [], [], []
        # text
        for pid, caption in self.txt_loader:
            caption = caption.to(device)
            with torch.no_grad():
                text_feat = model.encode_text(caption).cpu()
            qids.append(pid.view(-1)) # flatten 
            qfeats.append(text_feat)
        qids = torch.cat(qids, 0)
        qfeats = torch.cat(qfeats, 0)

        # refer
        for pid, caption in self.refer_loader:
            caption = caption.to(device)
            with torch.no_grad():
                refer_feat = model.encode_text(caption).cpu()
            refer_feats.append(refer_feat)
        refer_feats = torch.cat(refer_feats, 0)

        # image
        for pid, img in self.img_loader:
            img = img.to(device)
            with torch.no_grad():
                img_feat = model.encode_image(img).cpu()
                image_feats = model.encode_image1(img)
                token_features = model.img2text(img_feat.half().to(device))
                with autocast():
                    prompts = model.prompt_learner(token_features)
                    text_feature = model.text_encoder(prompts, model.prompt_learner.tokenized_prompts)
                    cross_x = model.cross_former(text_feature.unsqueeze(1), image_feats, image_feats)
                    cross_x_bn = model.bottleneck_proj(cross_x.squeeze(1))
            
            gids.append(pid.view(-1))  # flatten
            gfeats.append(img_feat)
            tp.append(text_feature.cpu().float())
            # cross.append(cross_x.squeeze(1).cpu().float())
            cross.append(cross_x_bn.cpu().float())

        gids = torch.cat(gids, 0)
        gfeats = torch.cat(gfeats, 0)
        tp = torch.cat(tp,0)
        cross = torch.cat(cross,0)

        return qfeats.cpu(), gfeats.cpu(), qids.cpu(), gids.cpu() , refer_feats.cpu(), tp.cpu(), cross.cpu()
    
    def _compute_embedding_tse(self, model):
        model = model.eval()
        device = next(model.parameters()).device

        qids, gids, qfeats, gfeats, refer_feats = [], [], [], [], []
        # text
        for pid, caption in self.txt_loader:
            caption = caption.to(device)
            with torch.no_grad():
                text_feat = model.encode_text_tse(caption).cpu()
            qids.append(pid.view(-1)) # flatten
            qfeats.append(text_feat)
        qids = torch.cat(qids, 0)
        qfeats = torch.cat(qfeats, 0)

        # refer
        for pid, caption in self.refer_loader:
            caption = caption.to(device)
            with torch.no_grad():
                refer_feat = model.encode_text_tse(caption).cpu()
            refer_feats.append(refer_feat)
        refer_feats = torch.cat(refer_feats, 0)

        # image
        for pid, img in self.img_loader:
            img = img.to(device)
            with torch.no_grad():
                img_feat = model.encode_image_tse(img).cpu()
            gids.append(pid.view(-1)) # flatten
            gfeats.append(img_feat)
        gids = torch.cat(gids, 0)
        gfeats = torch.cat(gfeats, 0)
        return qfeats.cpu(), gfeats.cpu(), qids.cpu(), gids.cpu(), refer_feats.cpu()


    def eval(self, model, i2t_metric=False):
        qfeats, gfeats, qids, gids, refer, gtps, gcorss = self._compute_embedding(model)
        qfeats = F.normalize(qfeats, p=2, dim=1) # text features
        gfeats = F.normalize(gfeats, p=2, dim=1) # image features
        gtps = F.normalize(gtps, p=2, dim=1)
        gcorss = F.normalize(gcorss, p=2, dim=1)
        refer = F.normalize(refer, p=2, dim=1) # refer features
        sims_bse = qfeats @ gfeats.t()
        gtp_sims = qfeats @ gtps.t()
        gcorss_sims = qfeats @ gcorss.t()

        vq_feats, vg_feats, _, _, vrefer = self._compute_embedding_tse(model)
        vq_feats = F.normalize(vq_feats, p=2, dim=1) # text features
        vg_feats = F.normalize(vg_feats, p=2, dim=1) # image features
        vrefer = F.normalize(vrefer, p=2, dim=1)  # refer features
        sims_tse = vq_feats@vg_feats.t()


        knc_k = 128
        knc_alpha = 0.75
        if self.args.dataset_name == "CUHK-PEDES":
            knc_k = 32
            knc_alpha = 0.6
        if self.args.dataset_name == "ICFG-PEDES":
            knc_k = 512
            knc_alpha = 0.8
        if self.args.dataset_name == "RSTPReid":
            knc_k = 32
            knc_alpha = 0.7
        reference_embeddings = qfeats.numpy()    # refer.numpy()
        # reference_embeddings = refer.numpy()  # refer.numpy()
        knc_retriever = kncRetriever(gfeats.shape[1], use_gpu=True, gpu_id=0)
        knc_ranker = kncRanker(knc_retriever, gfeats.numpy(), reference_embeddings, alternate_ks=knc_k,
                                alternate_weight=knc_alpha, batch_size=512, use_gpu=True, gpu_id=0)
        k = 10
        knc_scores1, knc_indices = knc_ranker.search(qfeats.numpy(), k)
        
        # ----------------------------------------------------------------------------------------------------------
        reference_embeddings = vq_feats.numpy()   # vrefer.numpy()
        # reference_embeddings = vrefer.numpy()  # vrefer.numpy()
        knc_retriever = kncRetriever(vg_feats.shape[1], use_gpu=True, gpu_id=0)
        knc_ranker = kncRanker(knc_retriever, vg_feats.numpy(), reference_embeddings, alternate_ks=knc_k,
                                alternate_weight=knc_alpha, batch_size=512, use_gpu=True, gpu_id=0)
        k = 10
        knc_scores2, knc_indices = knc_ranker.search(vq_feats.numpy(), k)


        sims_dict = {
            'BGE_knc': knc_scores1,
            'TSE_knc': knc_scores2,
            'BGE+TSE_knc(0.32)': 0.32 * knc_scores1 + 0.68 * knc_scores2,
        }

        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP","rSum"])

        top1 = 0

        for key in sims_dict.keys():
            sims = sims_dict[key]
            rs = get_metrics(sims, qids, gids, f'{key}-t2i',False)
            table.add_row(rs)
            if i2t_metric:
                i2t_cmc, i2t_mAP, i2t_mINP, _ = rank(similarity=sims.t(), q_pids=gids, g_pids=qids, max_rank=10, get_mAP=True)
                i2t_cmc, i2t_mAP, i2t_mINP = i2t_cmc.numpy(), i2t_mAP.numpy(), i2t_mINP.numpy()
                table.add_row(['i2t', i2t_cmc[0], i2t_cmc[4], i2t_cmc[9], i2t_mAP, i2t_mINP])

            top1 = max(top1,rs[1])

        table.custom_format["R1"] = lambda f, v: f"{v:.2f}"
        table.custom_format["R5"] = lambda f, v: f"{v:.2f}"
        table.custom_format["R10"] = lambda f, v: f"{v:.2f}"
        table.custom_format["mAP"] = lambda f, v: f"{v:.2f}"
        table.custom_format["mINP"] = lambda f, v: f"{v:.2f}"
        table.custom_format["RSum"] = lambda f, v: f"{v:.2f}"
        
        self.logger.info('\n' + str(table))
        self.logger.info('\n' + "best R1 = " + str(top1))

        return top1