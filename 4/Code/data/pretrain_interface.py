import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer, EsmForMaskedLM 

def get_pretrained_ESM(name):
    if name == "ESM35M":
        esm_dim = 480
        tokenizer = AutoTokenizer.from_pretrained("/model_zoom/transformers/models--facebook--esm2_t12_35M_UR50D")
        pretrain_model = EsmForMaskedLM.from_pretrained("/huyuqi/model_zoom/transformers/models--facebook--esm2_t12_35M_UR50D")# 480
    
    if name == "ESM650M":
        esm_dim = 1280
        tokenizer = AutoTokenizer.from_pretrained("/model_zoom/transformers/models--facebook--esm2_t33_650M_UR50D/snapshots/08e4846e537177426273712802403f7ba8261b6c")
        pretrain_model = EsmForMaskedLM.from_pretrained("/model_zoom/transformers/models--facebook--esm2_t33_650M_UR50D/snapshots/08e4846e537177426273712802403f7ba8261b6c") # 1280
    
    
    if name == "ESM35M_data1M":
        esm_dim = 480
        tokenizer = AutoTokenizer.from_pretrained("/model_zoom/transformers/models--facebook--esm2_t12_35M_UR50D")
        
        from PretrainESM_lightning.model import MInterface
        pretrain_args = OmegaConf.load("/xmyu/DiffSDS/PretrainESM_lightning/results/ESM35M/configs/10-09T07-29-05-project.yaml")
        pretrain_model = MInterface(**pretrain_args)
        ckpt = torch.load('/xmyu/DiffSDS/PretrainESM_lightning/results/ESM35M/checkpoints/best-epoch=13-val_seq_loss=2.604.pth')
        state_dict = {k.replace('_forward_module.', ''): v for k, v in ckpt.items()}
        pretrain_model.load_state_dict(state_dict)
    
    if name == "ESM35M_data1M_pad512":
        esm_dim = 480
        tokenizer = AutoTokenizer.from_pretrained("/model_zoom/transformers/models--facebook--esm2_t12_35M_UR50D")
        
        from PretrainESM_lightning.model import MInterface
        pretrain_args = OmegaConf.load("/xmyu/DiffSDS/PretrainESM_lightning/results/ESM35M_pad512/configs/10-09T08-08-05-project.yaml")
        pretrain_model = MInterface(**pretrain_args)
        ckpt = torch.load('/xmyu/DiffSDS/PretrainESM_lightning/results/ESM35M_pad512/checkpoints/best-epoch=13-val_seq_loss=2.604.pth')
        state_dict = {k.replace('_forward_module.', ''): v for k, v in ckpt.items()}
        pretrain_model.load_state_dict(state_dict)
        

    
    return esm_dim, tokenizer, pretrain_model


