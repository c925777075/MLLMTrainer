import torch
import numpy as np
from torch.utils.data import Sampler
from tqdm import tqdm
import math
import copy
import numpy as np
from multiprocessing import Pool
import multiprocessing as mp

class BalancedLengthDistributedSampler(Sampler):
    """分布式训练中基于样本长度均衡的采样器，确保每个GPU上的样本总长度接近。"""
    
    def __init__(self, dataset, batch_size_per_gpu, num_replicas=None, rank=None, 
                 shuffle=True, seed=0, drop_last=False, 
                 length_key='length'):
        """
        初始化采样器。
        
        参数:
            dataset: 数据集，需包含样本长度信息。
            num_replicas: 分布式训练的进程数（GPU数）。
            rank: 当前进程的排名。
            shuffle: 是否打乱数据。
            seed: 随机种子。
            drop_last: 是否丢弃剩余样本。
            length_key: 数据集中存储样本长度的键名。
        """
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("需要分布式环境但未找到分布式模块")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("需要分布式环境但未找到分布式模块")
            rank = torch.distributed.get_rank()
            
        self.dataset = dataset
        self.batch_size = batch_size_per_gpu
        
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last
        self.length_key = length_key

        # 获取所有样本的长度
        try:
            self.sample_lengths = self.dataset.lengths
        except:
            self.sample_lengths = self.dataset['lengths']
        assert self.sample_lengths is not None, "数据集必须包含样本长度信息"
        
        # 如果丢弃最后部分样本，调整总样本数
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            self.num_samples = math.floor(len(self.dataset) / self.num_replicas)
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        
    def _get_sample_lengths(self):
        """获取数据集中所有样本的长度"""
        lengths = []
        for i in tqdm(range(len(self.dataset)), total=len(self.dataset)):
            item = self.dataset[i]
            if isinstance(item, dict) and self.length_key in item:
                lengths.append(item[self.length_key])
            else:
                # 尝试其他可能的获取长度方式
                try:
                    lengths.append(len(item["input_ids"]))
                except:
                    raise ValueError(f"无法获取样本 {i} 的长度，请确保数据集包含长度信息")
        return np.array(lengths)
    
    def set_epoch(self, epoch):
        """设置当前训练轮次，用于生成不同的随机序列"""
        self.epoch = epoch
    
    def __iter__(self):
        """生成当前进程的样本索引"""
        if self.shuffle:
            # 结合epoch和seed生成随机种子，确保每轮训练的随机性
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        
        # 如果需要丢弃最后部分样本
        if self.drop_last:
            indices = indices[:self.total_size]
        else:
            # 填充到足够的长度
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        
        assert len(indices) == self.total_size
        
        # 按长度对样本进行分组，尝试均衡每个GPU上的总长度
        indices_with_length = [(idx, self.sample_lengths[idx]) for idx in indices]
        indices_with_length.sort(key=lambda x: x[1], reverse=True)  # 按长度降序排列
        
        # 初始化每个GPU的样本集合和总长度
        gpu_samples = [[] for _ in range(self.num_replicas)]
        gpu_lengths = [[] for _ in range(self.num_replicas)]
        gpu_total_lengths = [0] * self.num_replicas
        
        # 贪心算法：每次将样本分配给当前总长度最小的GPU
        for idx, length in indices_with_length:
            min_gpu_idx = np.argmin(gpu_total_lengths)
            gpu_samples[min_gpu_idx].append(idx)
            gpu_lengths[min_gpu_idx].append(length)
            gpu_total_lengths[min_gpu_idx] += length
        
        # 重排每个GPU样本，使得长度均匀分布
        new_gpu_samples = []
        for samples in gpu_samples:
            new_samples = []
            interval = int(len(samples) / self.batch_size)
            for i in range(interval):
                new_samples.extend(samples[i::interval])
            
            assert len(new_samples) == len(samples)
            new_gpu_samples.append(new_samples)
        gpu_samples = new_gpu_samples

        # 获取当前GPU的样本
        # shuffle
        min_num_sample = min([len(samples) for samples in gpu_samples])
        random_index = list(range(min_num_sample))
        np.random.seed(self.seed + self.epoch)
        np.random.shuffle(random_index)
        random_index = np.array(random_index, dtype=np.int64)
        all_new_gpu_samples = []
        for i in range(self.num_replicas):
            new_gpu_samples = np.array(gpu_samples[i][:min_num_sample])[random_index]
            new_gpu_samples = new_gpu_samples.tolist() + gpu_samples[i][min_num_sample:]
            all_new_gpu_samples.append(new_gpu_samples)
        return iter(all_new_gpu_samples[self.rank])
    
    def __len__(self):
        """返回当前进程的样本数量"""
        return self.num_samples