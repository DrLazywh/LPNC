import copy
from model import objectives
from .clip_model import Transformer, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights,tokenize
import torch
import torch.nn as nn
from .CrossEmbeddingLayer_tse import TexualEmbeddingLayer, VisualEmbeddingLayer
from .triplet_loss import TripletLoss
from .supcontrast import SupConLoss
from torch.cuda.amp import autocast


def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class IM2TEXT(nn.Module):
    def __init__(self, embed_dim=512, middle_dim=512, output_dim=512, n_layer=2, dropout=0.1):
        super().__init__()
        self.fc_out = nn.Linear(middle_dim, output_dim)
        layers = []
        dim = embed_dim
        for _ in range(n_layer):
            block = []
            block.append(nn.Linear(dim, middle_dim))
            block.append(nn.Dropout(dropout))
            block.append(nn.ReLU())
            dim = middle_dim
            layers.append(nn.Sequential(*block))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        # x  = self.bottleneck(x)
        for layer in self.layers:
            x = layer(x)
        return self.fc_out(x)

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        # ipdb.set_trace()
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND

        outputs = self.transformer([x])
        x = outputs[0]
        att = outputs[1]
        x = x.permute(1, 0, 2)  # LND -> NLD   # x,att
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        text_feature = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return text_feature



class PPL(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()
        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']
        self.tse_embed_dim = 4096


        if 'cid' in args.loss_names:
            self.num_classes = num_classes + 1
            self.classifier_bge = nn.Linear(self.embed_dim , self.num_classes)
            nn.init.normal_(self.classifier_bge.weight.data, std=0.001)
            nn.init.constant_(self.classifier_bge.bias.data, val=0.0)
            self.mlp_bge = nn.Sequential(nn.Linear(2 * self.embed_dim, self.embed_dim),nn.LayerNorm(self.embed_dim),nn.GELU())
            self.classifier_tse = nn.Linear(self.tse_embed_dim, self.num_classes)
            nn.init.normal_(self.classifier_tse.weight.data, std=0.001)
            nn.init.constant_(self.classifier_tse.bias.data, val=0.0)
            self.mlp_tse = nn.Sequential(nn.Linear(2 * self.tse_embed_dim, self.tse_embed_dim),nn.LayerNorm(self.tse_embed_dim),nn.GELU())
            self.classifier_id_bge = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier_id_bge.weight.data, std=0.001)
            nn.init.constant_(self.classifier_id_bge.bias.data, val=0.0)
            self.classifier_id_tse = nn.Linear(self.tse_embed_dim, self.num_classes)
            nn.init.normal_(self.classifier_id_tse.weight.data, std=0.001)
            nn.init.constant_(self.classifier_id_tse.bias.data, val=0.0)
            
        self.logit_scale = torch.ones([]) * (1 / args.temperature) 
        self.visul_emb_layer = VisualEmbeddingLayer(ratio=args.select_ratio)
        self.texual_emb_layer = TexualEmbeddingLayer(ratio=args.select_ratio)

        self.classifier_proj = nn.Linear(self.embed_dim, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)
        self.bottleneck_proj = nn.BatchNorm1d(self.embed_dim)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                self.embed_dim // 64,
                                                batch_first=True)
        self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                   layers=args.cmt_depth,
                                                   heads=self.embed_dim //
                                                         64)
        scale = self.cross_modal_transformer.width ** -0.5
        self.ln_pre_t = LayerNorm(self.embed_dim)
        self.ln_pre_i = LayerNorm(self.embed_dim)
        self.ln_post = LayerNorm(self.embed_dim)
        proj_std = scale * ((2 * self.cross_modal_transformer.layers) ** -0.5)
        attn_std = scale
        fc_std = (2 * self.cross_modal_transformer.width) ** -0.5
        for block in self.cross_modal_transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        # init cross attn
        nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
        nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

        self.text_encoder = TextEncoder(copy.deepcopy(self.base_model))
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.img2text = IM2TEXT(embed_dim=512,
                                middle_dim=512,
                                output_dim=512,
                                n_layer=2)
        dataset_name = args.dataset_name
        self.W = nn.Parameter(torch.eye(512))
        self.prompt_learner = PromptLearner(num_classes, dataset_name, self.base_model.dtype,
                                            self.base_model.token_embedding)
          
    def cross_former(self, q, k, v):
        x = self.cross_attn(
            self.ln_pre_t(q),
            self.ln_pre_i(k),
            self.ln_pre_i(v),
            need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x[0].unsqueeze(0)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)
        return x
    
    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')
    
    def encode_image(self, image):
        x, _ = self.base_model.encode_image(image)
        return x[:, 0, :].float()
      
    def encode_text(self, text):
        x, _ = self.base_model.encode_text(text.long())
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def encode_image_tse(self, image):
        x,atten_i = self.base_model.encode_image(image)
        i_tse_f = self.visul_emb_layer(x, atten_i)
        return i_tse_f.float()

    def encode_text_tse(self, text):
        x,atten_t = self.base_model.encode_text(text.long())
        t_tse_f = self.texual_emb_layer(x, text, atten_t)
        return t_tse_f.float()

    def forward(self, batch):
        ret = dict()
        device = "cuda"
        triplet = TripletLoss(margin=0.3)
        supcon = SupConLoss(device)

        if 'cid' in self.current_task:
            self.mlp_bge = self.mlp_bge.float()
            self.classifier_bge = self.classifier_bge.float()
            self.mlp_tse = self.mlp_tse.float()
            self.classifier_tse = self.classifier_tse.float()
        
        ret.update({'temperature': 1 / self.logit_scale})
        logit_scale = self.logit_scale
        images = batch['images']
        caption_ids = batch['caption_ids']
        image_feats, atten_i, text_feats, atten_t = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :].float()
        # i_feats = image_feats.float() # for CLIP ResNet visual model
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        i_tse_f = self.visul_emb_layer(image_feats, atten_i)
        t_tse_f = self.texual_emb_layer(text_feats, caption_ids, atten_t)

        if 'supid' in self.current_task:
            token_features = self.img2text(i_feats.half())
            with autocast():
                prompts = self.prompt_learner(token_features + t_feats @ self.W)
                text_feature = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)

            cross_x = self.cross_former(text_feature.unsqueeze(1), image_feats, image_feats)
            cross_x_bn = self.bottleneck_proj(cross_x.squeeze(1))
            cls_score = self.classifier_proj(cross_x_bn.half()).float()
            supcon_loss = supcon(i_feats, text_feature.float(), batch['pids'], batch['pids']) + supcon(text_feature.float(),i_feats,batch['pids'],batch['pids'])
            cross_id_loss = objectives.compute_id(cls_score, batch['pids']) 
            ret.update({'supid_loss': self.args.lambda1_weight * supcon_loss + self.args.lambda2_weight * cross_id_loss})

        if 'cid' in self.current_task:
            S = objectives.cosine_similarity_matrix(i_feats, t_feats)
            hard_negatives = objectives.sample_hard_negatives(S, batch['pids'])
            M = batch['pids'].max().item()
            new_labels = objectives.update_labels_for_negatives(batch['pids'], hard_negatives, M)
            all_pairs = objectives.create_sample_pairs(i_feats, t_feats, hard_negatives, new_labels, batch['pids'])
            all_cid_loss = 0
            ni_feats, nt_feats, nlabels = all_pairs
            z_feats1 = torch.cat([ni_feats.float(), nt_feats.float()], dim=1)
            z_feats2 = torch.cat([nt_feats.float(), ni_feats.float()], dim=1)
            z_feats1 = self.mlp_bge(z_feats1.float())
            z_feats2 = self.mlp_bge(z_feats2.float())
            cross_modal_logits1 = self.classifier_bge(z_feats1.float())
            cross_modal_logits2 = self.classifier_bge(z_feats2.float())
            device = cross_modal_logits1.device 
            nlabels = nlabels.to(device) 
            closs1 =  objectives.compute_cid(cross_modal_logits1, cross_modal_logits2,nlabels)
            
            S_ = objectives.cosine_similarity_matrix(i_tse_f, t_tse_f)
            hard_negatives_ = objectives.sample_hard_negatives(S_, batch['pids'])
            M_ = batch['pids'].max().item()
            new_labels_ = objectives.update_labels_for_negatives(batch['pids'], hard_negatives_, M_)
            all_pairs_ = objectives.create_sample_pairs(i_tse_f, t_tse_f, hard_negatives_, new_labels_, batch['pids'])
            all_cid_loss_ = 0
            ni_feats_, nt_feats_, nlabels_ = all_pairs_
            z_feats1_ = torch.cat([ni_feats_.float(), nt_feats_.float()], dim=1)
            z_feats2_ = torch.cat([nt_feats_.float(), ni_feats_.float()], dim=1)
            z_feats1_ = self.mlp_tse(z_feats1_.float())
            z_feats2_ = self.mlp_tse(z_feats2_.float())
            cross_modal_logits1_ = self.classifier_tse(z_feats1_.float())
            cross_modal_logits2_ = self.classifier_tse(z_feats2_.float())
            device_ = cross_modal_logits1_.device
            nlabels_ = nlabels_.to(device)
            closs2 =  objectives.compute_cid(cross_modal_logits1_, cross_modal_logits2_,nlabels_)

            image_logits = self.classifier_id_bge(i_feats.half()).float()
            text_logits = self.classifier_id_bge(t_feats.half()).float()
            closs3 = objectives.compute_id(image_logits, batch['pids']) + objectives.compute_id(text_logits, batch['pids'])
            
            image_logits_ = self.classifier_id_tse(i_tse_f.half()).float()
            text_logits_ = self.classifier_id_tse(t_tse_f.half()).float()
            closs4 = objectives.compute_id(image_logits_, batch['pids']) + objectives.compute_id(text_logits_, batch['pids'])
            ret.update({'cid_loss': closs1+closs2+closs3+closs4})


        if 'cotrl' in self.current_task:
            TAL_bge_loss = objectives.compute_TAL(i_feats, t_feats,batch['pids'],margin=self.args.margin,tau=self.args.tau)
            TAL_tse_loss = objectives.compute_TAL(i_tse_f, t_tse_f,batch['pids'],margin=self.args.margin,tau=self.args.tau)
            triloss_i, _, _ = triplet(i_feats, batch['pids'])
            triloss_t, _, _ = triplet(t_feats, batch['pids'])
            ret.update({'cotrl_loss': TAL_bge_loss + TAL_tse_loss + triloss_i + triloss_t})
            
        return ret


class PromptLearner(nn.Module):
    def __init__(self, num_class, dataset_name, dtype, token_embedding):
        super().__init__()
        ctx_init = "A photo of a X person"
        ctx_dim = 512
        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = 4
        tokenized_prompts = tokenize(ctx_init).cuda()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokenized_prompts = tokenized_prompts.to(device)
        token_embedding = token_embedding.to(device)
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + 1:, :])
        self.num_class = num_class
        self.token_ = token_embedding.to(device)
        self.dtype = dtype
    def forward(self, bias):
        b = bias.shape[0]
        prefix = self.token_prefix.expand(b, -1, -1)
        suffix = self.token_suffix.expand(b, -1, -1)
        bias = bias.unsqueeze(1)
        prompts = torch.cat(
            [
                prefix,
                bias,
                suffix,
            ],
            dim=1,
        )
        return prompts

def build_model(args, num_classes=11003):
    model = PPL(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model
