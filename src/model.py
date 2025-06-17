import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''
    '''https://github.com/jadore801120/attention-is-all-you-need-pytorch/'''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(1, 2))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class MyModel(nn.Module):
    def __init__(self, device, label_num, config):
        super(MyModel, self).__init__()
        self.device = device
        self.label_num = label_num

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config["feat_dim"]))
        self.bert = BertModel.from_pretrained("./mbert")
        self.dropout = nn.Dropout(p=config["dropout_rate"])
        self.friend_embedding = nn.Embedding(num_embeddings=label_num+1, embedding_dim=config["feat_dim"])
        self.q = nn.Linear(config["feat_dim"], config["feat_dim"])
        self.k = nn.Linear(config["feat_dim"], config["feat_dim"])
        self.v = nn.Linear(config["feat_dim"], config["feat_dim"])
        self.layer_norm = nn.LayerNorm(config["feat_dim"], eps=1e-6)
        self.fusion_attention = ScaledDotProductAttention(temperature=config["feat_dim"]**2)
        self.classifer = nn.Linear(config["feat_dim"], label_num)

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def forward(self, x):
        
        bsz = x["profile_id"].shape[0]

        # generating profile embedding via bert
        profile_id = x["profile_id"]
        profile_mask = x["profile_mask"]
        profile_type = x["profile_type"]

        profile_output = self.bert(
            input_ids=profile_id,
            attention_mask=profile_mask,
            token_type_ids=profile_type,
            return_dict=True
        )
        profile_pooler_output = profile_output.pooler_output # alt: profile_output.last_hidden_state
        profile_pooler_output = self.dropout(profile_pooler_output)
        profile_token = torch.unsqueeze(profile_pooler_output, dim=1) # [B, 1, dim]

        # generating social embedding via linear proj.
        friend_label_id = x["friend_label_id"]
        friend_label_mask = x["friend_label_mask"]
        # friend_profile = x["friend_name"]

        friend_embedding = self.friend_embedding.weight
        friend_label = friend_embedding[friend_label_id]
        friend_token = friend_label # + friend_profile
        friend_token = self.dropout(friend_token)

        # fuse features via self-attention
        cls_token = self.cls_token.expand(friend_token.shape[0], -1, -1)
        final_token = torch.cat([cls_token, profile_token, friend_token], dim=1)
        attention_mask = torch.cat([torch.ones(bsz, 1).to(self.device), torch.ones(bsz, 1).to(self.device), friend_label_mask], dim=1) # 插入cls token后：step1—更新mask向量；step2—与attn大小对齐
        attention_mask = attention_mask.view(attention_mask.shape[0], 1, attention_mask.shape[1]).expand(-1, attention_mask.shape[-1], -1)
        
        q, k, v = self.q(final_token), self.k(final_token), self.v(final_token)
        residual = q

        feat_fusion, _ = self.fusion_attention(q, k, v, mask=attention_mask)
        feat_fusion = self.dropout(feat_fusion)
        feat_fusion = feat_fusion
        feat_fusion = self.layer_norm(feat_fusion)
        cls_token = feat_fusion[:, 0, :]

        # obtain logits via classify head
        logit = self.classifer(cls_token)
        return logit

if __name__=="__main__":
    print("this is the neural network model")