import torch
from torch.utils.data import DataLoader
from src.engine.balanced_length_sampler import BalancedLengthDistributedSampler

def modify_trainer_for_balanced_sampler(trainer, dataset, length_key='length'):
    """
    修改训练器以使用基于样本长度均衡的分布式采样器。
    
    参数:
        trainer: 原始训练器实例。
        dataset: 训练数据集。
        length_key: 数据集中存储样本长度的键名。
    
    返回:
        修改后的训练器。
    """
    # 创建新的采样器
    sampler = BalancedLengthDistributedSampler(
        dataset,
        batch_size_per_gpu=trainer.args.train_batch_size,
        shuffle=True,
        length_key=length_key,
        drop_last=True,
    )
    
    # 获取当前的训练批次大小
    batch_size = trainer.args.train_batch_size
    
    # 创建新的数据加载器
    new_train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=trainer.data_collator,
        num_workers=trainer.args.dataloader_num_workers,
        pin_memory=trainer.args.dataloader_pin_memory,
        drop_last=True,
    )
    
    # 更新训练器的数据加载器
    trainer.get_train_dataloader = lambda: new_train_dataloader
    
    # 添加设置epoch的钩子，确保每个epoch使用不同的随机排列
    def _wrap_training_loop(original_training_loop):
        def new_training_loop(*args, **kwargs):
            # 在每个epoch开始前设置采样器的epoch
            for epoch in range(trainer.state.num_train_epochs):
                sampler.set_epoch(epoch)
                # 执行原始的训练循环
            return original_training_loop(*args, **kwargs)
        return new_training_loop
    
    # 包装训练循环函数
    trainer.training_step = _wrap_training_loop(trainer.training_step)
    
    return trainer