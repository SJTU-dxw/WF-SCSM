"""Persistent, model-independent weekly splits and lazy HDF5 features."""
from pathlib import Path
import hashlib
import json
import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]

def prepare_split(args):
    path = Path(args.dataset).resolve()
    stat = path.stat()
    website_count = getattr(args, 'website_count', None)
    spec = dict(dataset=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                reference_end=args.reference_end, min_trace_length=args.min_trace_length,
                min_reference_samples=args.min_reference_samples, min_week_samples=args.min_week_samples,
                test_per_class=args.test_per_class, split_seed=args.split_seed,
                website_count=website_count)
    key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:20]
    directory = Path(args.split_dir) / key
    manifest = directory / 'manifest.json'
    if manifest.exists():
        result = json.loads(manifest.read_text())
        if result['spec'] != spec:
            raise ValueError('Split specification mismatch')
        print(f"Eligible websites before top-count selection: {result['eligible_count']}",flush=True)
        print(f"Selected websites after top-count selection: {len(result['classes'])}",flush=True)
        return directory, result
    with h5py.File(path, 'r') as f:
        if not f.attrs.get('complete', True):
            raise ValueError('HDF5 conversion is incomplete')
        print('Reading HDF5 day/labels/lengths metadata...', flush=True)
        days, labels, lengths = (f[n][:] for n in ('day','labels','lengths'))
    valid = lengths >= args.min_trace_length
    weeks = [(s, min(s+6, int(days.max()))) for s in range(args.reference_end+1, int(days.max())+1, 7)]
    if not weeks:
        raise ValueError('No post-pretraining weeks')
    def qualified(mask, minimum):
        values, counts = np.unique(labels[valid & mask], return_counts=True)
        return set(values[counts >= minimum].tolist())
    sites = qualified(days <= args.reference_end, args.min_reference_samples)
    for start,end in weeks:
        print(f'Class filtering: days {start}-{end}', flush=True)
        sites &= qualified((days>=start)&(days<=end), args.min_week_samples)
    sites = sorted(sites)
    if len(sites)<2:
        raise ValueError('Fewer than two eligible classes; review class thresholds')
    eligible_count=len(sites)
    print(f'Eligible websites before top-count selection: {eligible_count}',flush=True)
    if website_count is not None:
        if website_count>eligible_count:
            raise ValueError(f'website-count {website_count} exceeds {eligible_count} eligible websites')
        eligible_labels=labels[valid & np.isin(labels,sites)]
        values,counts=np.unique(eligible_labels,return_counts=True)
        totals=dict(zip(values.tolist(),counts.tolist()))
        sites=sorted(sorted(sites,key=lambda site:(-totals[site],site))[:website_count])
    print(f'Selected websites after top-count selection: {len(sites)}',flush=True)
    if args.min_week_samples <= args.test_per_class:
        raise ValueError('min-week-samples must exceed test-per-class')
    print(f'Eligible classes: {len(sites)}; reserving fixed test samples', flush=True)
    arrays = {}; tests=[]
    eligible = valid & np.isin(labels, sites)
    for w,(start,end) in enumerate(weeks,1):
        pool=[]
        week_indices=np.flatnonzero(eligible & (days>=start) & (days<=end))
        order=np.argsort(labels[week_indices],kind='stable')
        week_indices=week_indices[order]
        _,starts,counts=np.unique(labels[week_indices],return_index=True,return_counts=True)
        for y,(begin,count) in enumerate(zip(starts,counts)):
            indices=week_indices[begin:begin+count]
            rng=np.random.default_rng(np.random.SeedSequence([args.split_seed,w,y]))
            indices=rng.permutation(indices)
            records=np.column_stack((indices,np.full(len(indices),y),np.full(len(indices),w)))
            tests.extend(records[:args.test_per_class])
            pool.extend(records[args.test_per_class:])
        arrays[f'pool_{w}']=np.asarray(pool,dtype=np.int64)
    arrays['test']=np.asarray(tests,dtype=np.int64)
    result=dict(spec=spec, eligible_count=eligible_count,
                classes=[s.decode('utf-8') for s in sites], weeks=[list(w) for w in weeks],
                test_count=len(tests))
    directory.mkdir(parents=True,exist_ok=True)
    # Serialize creators; never overwrite a published split.
    lock=directory/'creating.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    try:
        if not manifest.exists():
            np.savez_compressed(directory/'indices.npz',**arrays)
            temp=directory/'manifest.tmp'; temp.write_text(json.dumps(result,indent=2)); temp.replace(manifest)
    finally:
        os.close(fd); lock.unlink()
    return directory,result

def sample_train(pool,k,seed):
    rows=[]
    for label in np.unique(pool[:,1]):
        candidates=pool[pool[:,1]==label]
        if len(candidates)<k:
            raise ValueError(f'class {label}: only {len(candidates)} training samples remain, requested k={k}')
        rng=np.random.default_rng(np.random.SeedSequence([seed,int(candidates[0,2]),int(label)]))
        rows.extend(candidates[rng.permutation(len(candidates))[:k]])
    return np.asarray(rows,dtype=np.int64)

class WeeklyDataset(Dataset):
    def __init__(self,path,records,method,config,train=False):
        self.path=str(path); self.records=records; self.method=method; self.config=config; self.file=None
        self.train=train
        self.delegate=None
        if method in ('swallow-origin','swallow-single','netclr','traverse','countmamba'):
            from feature_similarity.evaluate import TraceDataset
            kind={'scsm-single':'scsm','countmamba':'scsm','swallow-origin':'swallow','swallow-single':'swallow'}.get(method,method)
            self.delegate=TraceDataset(Path(path),[(int(w),int(i),b'') for i,y,w in records],kind,config)
    def __len__(self): return len(self.records)
    def __getstate__(self):
        state=self.__dict__.copy(); state['file']=None; return state
    def _scsm_feature(self,row,slot):
        if self.file is None: self.file=h5py.File(self.path,'r',swmr=True)
        length=int(self.file['lengths'][row])
        directions=np.asarray(self.file['directions'][row,:length],dtype=np.float32)
        times=np.asarray(self.file['times'][row,:length],dtype=np.float64)
        times-=times[0]; times[0]=1e-6
        matrix_length=int(self.config['max_matrix_length'])
        maximum_cells=int(self.config['maximum_cell_number'])
        feature=np.zeros((2*(maximum_cells+2),matrix_length),dtype=np.float32)
        columns=np.floor(times/slot).astype(np.int64).clip(0,matrix_length-1)
        rows=(np.sign(directions)>0).astype(np.int64)
        np.add.at(feature,(rows,columns),1)
        unique,starts=np.unique(columns,return_index=True)
        feature[2*maximum_cells+2,unique[1:]]=np.diff(unique)
        threshold=slot*float(self.config['time_interval_threshold'])
        for group,begin in enumerate(starts):
            finish=starts[group+1] if group+1<len(starts) else len(times)
            feature[2*maximum_cells+3,unique[group]]=1+np.sum(
                np.diff(times[begin:finish])>threshold
            )
        if self.config.get('log_transform',True): feature=np.log1p(feature)
        return torch.from_numpy(feature[None]),torch.tensor(int(unique[-1]),dtype=torch.float32)
    def __getitem__(self,index):
        row,y,week=self.records[index]
        if self.method=='scsm-single':
            matrix_length=int(self.config['max_matrix_length'])
            if self.train and self.config.get('random_slot',False):
                slot=np.random.uniform(self.config['min_slot_time'],self.config['max_slot_time'])
                feature,last=self._scsm_feature(row,slot)
                return feature,last,int(y)
            strategy=self.config.get('test_slot_strategy','fixed')
            if strategy=='ensemble':
                slots=np.linspace(
                    self.config['min_slot_time'],self.config['max_slot_time'],
                    int(self.config['test_slot_count']),
                )
                views=[self._scsm_feature(row,slot) for slot in slots]
                return torch.stack([view[0] for view in views]),torch.stack([view[1] for view in views]),int(y)
            if strategy=='adaptive':
                if self.file is None: self.file=h5py.File(self.path,'r',swmr=True)
                length=int(self.file['lengths'][row])
                times=np.asarray(self.file['times'][row,:length],dtype=np.float64)
                load_time=float(times[-1]-times[0])
                slot=np.clip(
                    self.config['test_slot_multiplier']*load_time/matrix_length,
                    self.config['min_slot_time'],self.config['max_slot_time'],
                )
            else:
                slot=float(self.config['maximum_load_time'])/matrix_length
            feature,last=self._scsm_feature(row,slot)
            return feature,last,int(y)
        if self.delegate is not None:
            data=self.delegate[index]
            if self.method=='traverse': return data[0],int(y)
            x,last=data[:2]
            if self.method in ('swallow-origin','swallow-single'): x=x.unsqueeze(0)
            return x,last,int(y)
        if self.file is None: self.file=h5py.File(self.path,'r',swmr=True)
        length=min(int(self.file['lengths'][row]),self.file['directions'].shape[1])
        d=np.asarray(self.file['directions'][row,:length],dtype=np.float32)
        t=np.asarray(self.file['times'][row,:length],dtype=np.float64)
        t=t-t[0]; t[0]=1e-6
        def pad(x,n): return np.pad(x[:n],(0,max(0,n-len(x))))
        n=self.config['length']
        if self.method=='ares':
            from .vendor.features import process_MTAF
            x=process_MTAF(0,d*t*1000,20,n)[1]
        elif self.method=='rf':
            from .vendor.features import process_TAM
            x=process_TAM(0,d*t,80,n)[1][None]
        elif self.method=='varcnn': x=np.stack((pad(d,n),pad(np.maximum(np.diff(t),0),n)))
        else: x=pad(d*t if self.method=='tiktok' else d,n)[None]
        return torch.tensor(x,dtype=torch.float32),torch.tensor(0.),int(y)

class TextCollator:
    def __init__(self,tokenizer,config):
        from feature_similarity.evaluate import TraverseCollator
        self.base=TraverseCollator(tokenizer,config['max_length'],config['prompt_prefix'],config['prompt_suffix'])
    def __call__(self,samples):
        data,mask,_,_,_=self.base([(text,0,b'',i) for i,(text,y) in enumerate(samples)])
        return data,mask,torch.tensor([y for _,y in samples])
