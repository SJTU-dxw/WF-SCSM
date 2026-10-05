"""Model registry. Optional dependencies are imported only when needed."""
import importlib
import json
from pathlib import Path
import torch
from torch import nn
from .data import ROOT

# epochs, batch size, learning rate, optimizer, weight decay, input length
DEFAULTS={
 'awf':(200,256,8e-4,'RMSprop',0.,10000),
 'tmwf':(200,80,5e-4,'Adam',0.,10000),
 'ares':(200,512,2e-3,'AdamW',.01,8000),
 'df':(200,128,2e-3,'Adamax',0.,10000),
 'tiktok':(200,128,2e-3,'Adamax',0.,10000),
 'varcnn':(200,50,1e-3,'Adam',0.,10000),
 'rf':(200,200,5e-4,'Adam',0.,2700),
 'countmamba':(200,200,2e-3,'AdamW',.05,2700),
 'scsm-single':(200,128,1e-4,'AdamW',.05,2700),
 'netclr':(200,128,1e-4,'Adam',0.,10000),
 'swallow-origin':(200,128,1e-4,'Adam',0.,1000),
 'swallow-single':(200,128,1e-4,'Adam',0.,2700),
 'traverse':(100,2,2e-5,'AdamW',.01,512),
}
NAMES=dict(awf='AWF',tmwf='TMWF',ares='ARES',df='DF',tiktok='TikTok',varcnn='VarCNN',rf='RF')

class Classifier(nn.Module):
    def __init__(self,encoder,dim,classes,method):
        super().__init__(); self.encoder=encoder; self.method=method
        if method=='traverse':
            self.head=nn.Sequential(
                nn.Linear(dim,1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024,classes),
            )
        else:
            self.head=nn.Linear(dim,classes)
        self.freeze=False
    def train(self,mode=True):
        super().train(mode)
        if self.freeze: self.encoder.eval()
        return self
    def forward(self,x,extra):
        if self.method=='scsm-single': features=self.encoder(x,extra)
        elif self.method=='traverse':
            from feature_similarity.evaluate import linear_position_pool
            # Call the decoder directly: avoid allocating [B,L,vocabulary] LM logits.
            base=self.encoder.model.get_base_model().model
            hidden=base(**x,return_dict=True,use_cache=False).last_hidden_state
            features=linear_position_pool(hidden,extra)
        else: features=self.encoder(x).flatten(1)
        return self.head(features.float())

class Baseline(nn.Module):
    def __init__(self,model): super().__init__(); self.model=model
    def forward(self,x,extra): return self.model(x)

def build_model(method,classes,device,model_dir=None,freeze=False):
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and resolved_device.index is not None:
        torch.cuda.set_device(resolved_device)
    tokenizer=None
    if method in NAMES:
        name=NAMES[method]
        input_length=DEFAULTS[method][-1]
        model_class=getattr(importlib.import_module(f'finetune.vendor.{name}'),name)
        sequence_methods={'awf','tmwf','df','tiktok','varcnn'}
        model_args={'num_classes':classes}
        if method in sequence_methods: model_args['input_length']=input_length
        model=Baseline(model_class(**model_args))
        config={'length':input_length}; source=None
    elif method=='countmamba':
        from model.scsm import SCSM
        class CountMamba(SCSM):
            def __init__(self,c):
                super().__init__(c); self.fc_norm=nn.LayerNorm(self.output_dim); self.fc=nn.Linear(self.output_dim,classes)
            def forward(self,x,idx):
                z=self.local_model(self.patch_embed(x.float()))+self.pos_embed[:,1:]
                z=torch.cat(((self.cls_token+self.pos_embed[:,:1]).expand(len(x),-1,-1),z),1)
                for block,drop in zip(self.blocks,self.drop_paths): z=z+drop(block(z))
                z=self.fc_norm(z)[:,1:]
                ends=torch.floor(idx/6).long().clamp(0,z.shape[1]-1)
                return self.fc(torch.stack([r[:int(end)+1].mean(0) for r,end in zip(z,ends)]))
        config=dict(max_matrix_length=DEFAULTS[method][-1],maximum_cell_number=2,embedding_dim=256,depth=3,
                    drop_path_rate=.2,simple_mode=True,kernel_size=5,maximum_load_time=120,
                    time_interval_threshold=.1,log_transform=True)
        model=CountMamba(config); source=None
    else:
        from feature_similarity.evaluate import (load_model,SWALLOW_ORIGIN_CIF_CONFIG,SWALLOW_SINGLE_CIF_CONFIG,
            TRAVERSE_MAX_BURSTS,TRAVERSE_PROMPT_PREFIX,TRAVERSE_PROMPT_SUFFIX)
        folder=Path(model_dir) if model_dir else ROOT/'output/pretrain/GTT_dataset'/ {'scsm-single':'scsm_single'}.get(method,method)
        pretrained=json.loads((folder/'config.json').read_text())
        expected={'scsm-single':'scsm','swallow-origin':'swallow','swallow-single':'swallow'}.get(method,method)
        if pretrained['method']!=expected: raise ValueError('Checkpoint method mismatch')
        if method=='scsm-single' and not pretrained['model'].get('simple_mode'):
            raise ValueError('Expected SCSM single checkpoint')
        if method in ('swallow-origin','swallow-single'):
            expected_dataset={'swallow-origin':'gtt23-swallow-origin','swallow-single':'gtt23-swallow-single'}[method]
            if Path(pretrained['dataset']['path']).name.lower()!=expected_dataset:
                raise ValueError(f'Expected Swallow {method.removeprefix("swallow-")} checkpoint')
        if method=='traverse':
            from transformers import AutoTokenizer
            tokenizer_path=Path(pretrained['model']['pretrained_model'])
            if not tokenizer_path.exists(): tokenizer_path=ROOT/'pretrain'/tokenizer_path.name
            tokenizer=AutoTokenizer.from_pretrained(tokenizer_path,use_fast=True)
            if tokenizer.mask_token_id is None: tokenizer.add_special_tokens({'mask_token':'<|mask|>'})
            if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
            config=dict(max_bursts=TRAVERSE_MAX_BURSTS,prompt_prefix=TRAVERSE_PROMPT_PREFIX,
                        prompt_suffix=TRAVERSE_PROMPT_SUFFIX,max_length=pretrained['dataset']['max_length'])
        elif method=='swallow-origin': config=dict(SWALLOW_ORIGIN_CIF_CONFIG)
        elif method=='swallow-single': config=dict(SWALLOW_SINGLE_CIF_CONFIG)
        else: config={**pretrained['model'],**pretrained.get('augmentation',{})}
        encoder,source=load_model(pretrained,folder,torch.device('cpu'),tokenizer)
        if method=='traverse':
            dim=encoder.model.config.hidden_size
            encoder.model.enable_input_require_grads()
            config['classifier_hidden_dim']=1024
        elif method in ('swallow-origin','swallow-single'): dim=2048 if pretrained['model']['name']=='resnet50' else 512
        else: dim=encoder.output_dim
        model=Classifier(encoder,dim,classes,method)
        if freeze: model.encoder.requires_grad_(False); model.freeze=True
        config['pretrained_config']=pretrained
    if method in ('scsm-single','countmamba') and torch.device(device).type!='cuda':
        raise ValueError(f'{method} requires CUDA for the Mamba2 kernels')
    return model.to(device),config,tokenizer,str(source) if source else None
