import inspect
from tqdm import tqdm
import pickle as pkl
import pytorch_lightning as pl
# from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader as GeometricDataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from data.datasets import ECDataset, FoldDataset, FuncDataset, GODataset
import torch
import numpy as np
import torch.nn.functional as F
from torch_scatter import scatter_sum
from transformers import AutoTokenizer, EsmForMaskedLM 
from model.PretrainGearNet import PretrainGearNet_Model
import lmdb
import os.path as osp
from .pretrain_interface import get_pretrained_ESM, get_pretrained_VQ

class PretrainFeat:
    def __init__(self, pad=768):
        self.pad = pad
        aa = "ACDEFGHIKLMNPQRSTVWYX"
        self.id_to_aa = {}
        for i in range(0, 21):
            self.id_to_aa[i] = aa[i]
        
        from transformers import AutoTokenizer
        self.ESM_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir="/model_zoom/transformers")
        
        from PretrainStage2_lightning.data.convertor import Converter
        self.data_converter = Converter()

    def get_seqs_angles(self, data):
        device = data.x.device
        seqs_list, angles_list, attn_mask_list = [], [], []
        
        residue_num = scatter_sum(torch.ones_like(data.batch), data.batch)
        self.pad = residue_num.max()
        # assert residue_num.max() <= 1024

        for i in data.batch.unique():
            mask = data.batch == i
            seq = [self.id_to_aa[aa_id.item()] for aa_id in data.x[mask]]
            seq = self.ESM_tokenizer.encode("".join(seq), add_special_tokens=False)
            seq = torch.tensor(seq, device=device).reshape(-1,1)
            coord = data.pos[mask].cpu().numpy()
            L = coord.shape[0]
            
            angles = self.data_converter.coord2angle(coord)
            angles = torch.nan_to_num(angles, nan=0.0, posinf=0.0, neginf=0.0)
            angles = angles.to(device)

            # self.pad = 1024
            seq = F.pad(
                seq,
                (0,0,0, self.pad - seq.shape[0]),
                mode="constant",
                value=self.ESM_tokenizer.pad_token_id,
            )
            
            angles = F.pad(
                angles,
                (0, 0, 0, self.pad - angles.shape[0])
            )
        
            attn_mask = torch.zeros(size=(self.pad,), device=device)
            attn_mask[:L] = 1.0
    
            seqs_list.append(seq)
            angles_list.append(angles)
            attn_mask_list.append(attn_mask)

        seqs = torch.stack(seqs_list, dim=0)
        angles = torch.stack(angles_list, dim=0)
        attn_mask = torch.stack(attn_mask_list, dim=0)
        return seqs, angles, attn_mask

class MyDataLoader(GeometricDataLoader):
    def __init__(self, pretrain_model_type, batch_size=64, num_workers=8,  esm_version="ESM35M", *args, **kwargs):
        super().__init__(batch_size=batch_size, num_workers=num_workers, *args, **kwargs)
        self.pretrain_featurizer = PretrainFeat() 
        self.esm_version = esm_version
        self.pretrain_device = 'cuda:0'
        self.pretrain_model_type = pretrain_model_type
        self.esm_dim, self.tokenizer, self.pretrain_esm_model = get_pretrained_ESM(esm_version)
        self.pretrain_esm_model = self.pretrain_esm_model.to(self.pretrain_device)
    
        self.stream = torch.cuda.Stream(
            self.pretrain_device
        )  # create a new cuda stream in each process
        
        self.memory = {}

    
    def get_esm_embedding(self, seqs, attn_mask):
        if self.esm_version in ["ESM35M_data1M", "ESM35M_data1M_pad512", "ESM35M_data1M_pad512_flash"]:
            outputs = self.pretrain_esm_model.model(input_ids=seqs[:,:,0], attention_mask=attn_mask)
            pretrain_embedding = outputs.hidden_states
            pretrain_embedding = pretrain_embedding.reshape(-1,self.esm_dim)[attn_mask.view(-1)==1]
            pretrain_embedding = pretrain_embedding.cpu()
        else:
            outputs = self.pretrain_esm_model(input_ids=seqs[:,:,0], attention_mask=attn_mask)
            pretrain_embedding = outputs.hidden_states
            pretrain_embedding = pretrain_embedding.reshape(-1,self.esm_dim)[attn_mask.view(-1)==1]
            pretrain_embedding = pretrain_embedding.cpu()
        return pretrain_embedding



            
    
    def __iter__(self):
        for batch in super().__iter__():
            if all([name in self.memory for name in batch.protein_name]):
                batch.pretrain_embedding = torch.cat([self.memory[one] for one in batch.protein_name], dim=0)
                assert batch.pretrain_embedding.shape[0] == batch.x.shape[0]
            else:
                with torch.no_grad():
                    seqs, angles, attn_mask = self.pretrain_featurizer.get_seqs_angles(batch.to(self.pretrain_device))
                    pretrain_embedding = [torch.zeros(size=(batch.batch.shape[0], 0))]
                    
                    if "esm" in self.pretrain_model_type:
                        new_embedding = self.get_esm_embedding(seqs, attn_mask)
                        pretrain_embedding.append(new_embedding)
                    
                    if "vq" in self.pretrain_model_type:
                        new_embedding = self.get_vq_embedding(seqs, angles, attn_mask)
                        pretrain_embedding.append(new_embedding)
                    
                    if "gear" in self.pretrain_model_type:
                        new_embedding = self.get_gearnet_embedding(batch, seqs, attn_mask)
                        pretrain_embedding.append(new_embedding)

                    pretrain_embedding = torch.cat(pretrain_embedding, dim=1)
            
                    batch.pretrain_embedding = pretrain_embedding
                    assert batch.pretrain_embedding.shape[0] == batch.x.shape[0]
                    
                    for bid, name in enumerate(batch.protein_name):
                        mask = batch.batch.cpu()==bid
                        self.memory[name] = pretrain_embedding[mask]


            if type(batch.y)==list:
                batch.y = torch.from_numpy(np.stack(batch.y, axis=0))
            with torch.cuda.stream(self.stream):
                batch = batch.to(device=self.pretrain_device, non_blocking=True)
                
            yield batch
            
           
            
            

class DInterface(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.dataset = kwargs['dataset'] 
        self.batch_size = kwargs['batch_size']
        print("batch_size", self.batch_size)
        self.load_data_module()
        if self.dataset == 'ec':
            self.trainset, self.valset, self.testset = None, None, None
        
        if self.dataset == 'fold':
            self.trainset, self.valset, self.test_fold, self.test_family, self.test_super = None, None, None, None, None
        
        if self.dataset == 'func':
            self.trainset, self.valset, self.testset = None, None, None
        
        if self.dataset == 'go':
            self.trainset, self.valset, self.test_dataset = None, None, None

    
    def setup(self, stage=None):
        if self.dataset == 'ec':
            # Assign train/val datasets for use in dataloaders
            if stage == 'fit' or stage is None:
                if self.trainset is None:
                    self.trainset = self.instancialize(split = 'train')
                
                if self.valset is None:
                    # self.valset = self.instancialize(split='valid')
                    self.valset = self.instancialize(split='test')

            # Assign test dataset for use in dataloader(s)
            if stage == 'test' or stage is None:
                if self.testset is None:
                    self.testset = self.instancialize(split='test')
        
        if self.dataset == 'fold':
            # Assign train/val datasets for use in dataloaders
            if stage == 'fit' or stage is None:
                if self.trainset is None:
                    self.trainset = self.instancialize(split = 'training')
                
                if self.valset is None:
                    # self.valset = self.instancialize(split='validation')
                    self.valset = self.instancialize(split='test_fold')
            
            if stage == 'test' or stage is None:
                if self.test_fold is None:
                    self.test_fold = self.instancialize(split='test_fold')
                
                if self.test_family is None:
                    self.test_family = self.instancialize(split='test_family')
                
                if self.test_super is None:
                    self.test_super = self.instancialize(split='test_superfamily')
            
        if self.dataset == 'func':
            # Assign train/val datasets for use in dataloaders
            if stage == 'fit' or stage is None:
                if self.trainset is None:
                    self.trainset = self.instancialize(split = 'training')
                
                if self.valset is None:
                    # self.valset = self.instancialize(split='validation')
                    self.valset = self.instancialize(split='testing')
            
            if stage == 'test' or stage is None:
                if self.testset is None:
                    self.testset = self.instancialize(split='testing')
        
        if self.dataset == 'go':
            # Assign train/val datasets for use in dataloaders
            if stage == 'fit' or stage is None:
                if self.trainset is None:
                    self.trainset = self.instancialize(split = 'train')
                
                if self.valset is None:
                    # self.valset = self.instancialize(split='valid')
                    self.valset = self.instancialize(split='test')
            
            if stage == 'test' or stage is None:
                self.test_dataset = self.instancialize(split='test')

    def train_dataloader(self, db=None, preprocess=False):
        return self.instancialize_module(MyDataLoader, dataset=self.trainset, shuffle=True, prefetch_factor=3)
    

    def val_dataloader(self, db=None, preprocess=False):
        return self.instancialize_module(MyDataLoader, dataset=self.valset,  shuffle=False, prefetch_factor=3)
    

    def test_dataloader(self, db=None, preprocess=False):
        if self.dataset == 'ec':
            return self.instancialize_module(MyDataLoader, dataset=self.testset, shuffle=False)

        if self.dataset == 'fold':
            return self.instancialize_module(MyDataLoader, dataset=self.test_fold, shuffle=False),\
            self.instancialize_module(MyDataLoader, dataset=self.test_family, shuffle=False),\
            self.instancialize_module(MyDataLoader, dataset=self.test_super, shuffle=False)
        
        
        if self.dataset == 'func':
            return self.instancialize_module(MyDataLoader, dataset=self.testset,  shuffle=False)

        if self.dataset == 'go':
            return self.instancialize_module(MyDataLoader, dataset=self.test_dataset,  shuffle=False)


    def load_data_module(self):
        name = self.dataset
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        if camel_name == 'Ec':
            self.data_module = ECDataset
        elif camel_name == 'Fold':
            self.data_module = FoldDataset
        elif camel_name == 'Func':
            self.data_module = FuncDataset
        elif camel_name == 'Go':
            self.data_module = GODataset

    def instancialize(self, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.kwargs.
        """
        class_args =  list(inspect.signature(self.data_module.__init__).parameters)[1:]
        inkeys = self.kwargs.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.kwargs[arg]
        args1.update(other_args)
        return self.data_module(**args1)
    
    def instancialize_module(self, module, **other_args):
        class_args =  list(inspect.signature(module.__init__).parameters)[1:]
        inkeys = self.kwargs.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.kwargs[arg]
        args1.update(other_args)
        return module(**args1)
