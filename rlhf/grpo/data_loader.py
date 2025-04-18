from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader, DistributedSampler

class GSM8KDataLoader:
    def __init__(self, config):
        self.tokenizer = AutoTokenizer.from_pretrained(config['model']['name'], add_special_tokens=True)
        self.config = config

        # 分布式环境检测
        self.distributed = torch.distributed.is_initialized()
        if self.distributed:
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

    @staticmethod
    def collate_fn(batch):
        return {
            'input_ids': torch.stack([torch.tensor(item['input_ids']) for item in batch]),
            'attention_mask': torch.stack([torch.tensor(item['attention_mask']) for item in batch]),
        }

    def preprocess(self, examples):
        """数据预处理"""
        inputs = self.tokenizer(
            examples['question'],
            padding='max_length',
            truncation=True,
            max_length=self.config['dataset']['max_length'],
            return_tensors='pt'
        )
        
        # 确保返回的是Python原生类型而非张量
        return {
            'input_ids': inputs['input_ids'].tolist(),  # 转换为列表
            'attention_mask': inputs['attention_mask'].tolist()
        }

    def load_dataset(self):
        """加载数据集，自动适应单卡/分布式训练"""
        dataset = load_dataset(self.config['dataset']['gsm8k_dir'], name='main')  # 或 'socratic' 根据需求选择

        # 数据预处理
        dataset = dataset.map(
            self.preprocess,
            batched=True,
            remove_columns = ['question', 'answer']
        )

        # 创建采样器（分布式或普通）
        sampler = DistributedSampler(
            dataset['train'],
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True
        ) if self.distributed else None

        # 创建 DataLoader
        train_loader = DataLoader(
            dataset['train'],
            batch_size=self.config['dataset']['batch_size'],
            sampler=sampler,
            shuffle=(sampler is None),  # 如果没有采样器，就需要 shuffle
            num_workers = self.config['dataset']['num_workers'],
            collate_fn=self.collate_fn
        )
        val_loader = DataLoader(
            dataset['test'],
            batch_size=self.config['dataset']['batch_size'],
            num_workers = self.config['dataset']['num_workers'],
            collate_fn=self.collate_fn
        )
        return train_loader, val_loader

    def preprocess(self, examples):
        """数据预处理"""
        inputs = self.tokenizer(
            examples['question'],
            padding='max_length',
            truncation=True, # 自动截断超过 max_length 的文本
            max_length = self.config['dataset']['max_length'],
            return_tensors = 'pt'
        )

        # 修改返回格式为字典而非张量
        return {
            'input_ids': inputs['input_ids'],  # 去除batch维度
            'attention_mask': inputs['attention_mask']
        }

########### test #############
import pytest
import yaml
import torch

@pytest.fixture
def config():
    """测试配置fixture"""
    with open('d:/Document/学习/algorithm/rlhf/grpo/configs/config_test.yaml') as f:
        return yaml.safe_load(f)

def test_tokenizer_initialization(config):
    """测试tokenizer初始化"""
    loader = GSM8KDataLoader(config)
    assert loader.tokenizer is not None
    assert isinstance(loader.tokenizer, AutoTokenizer)
    assert loader.config == config

def test_distributed_defaults(config):
    """测试单机模式默认值"""
    loader = GSM8KDataLoader(config)
    assert loader.rank == 0
    assert loader.world_size == 1

def test_dataset_loading(config):
    """测试数据集加载"""
    loader = GSM8KDataLoader(config)
    train_loader, val_loader = loader.load_dataset()
    
    # 验证DataLoader对象
    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)
    
    # 验证批次数据
    train_batch = next(iter(train_loader))
    assert isinstance(train_batch, dict)
    assert 'input_ids' in train_batch
    assert 'attention_mask' in train_batch

def test_batch_shapes(config):
    """测试批次形状"""
    loader = GSM8KDataLoader(config)
    train_loader, _ = loader.load_dataset()
    batch = next(iter(train_loader))
    
    # 验证形状
    assert batch['input_ids'].shape == (
        config['dataset']['batch_size'],
        config['dataset']['max_length']
    )
    assert batch['attention_mask'].shape == (
        config['dataset']['batch_size'],
        config['dataset']['max_length']
    )

def test_preprocessing(config):
    """测试数据预处理"""
    loader = GSM8KDataLoader(config)
    test_samples = {
        'question': ['What is 2+2?', 'Calculate 3*5'],
        'answer': ['4', '15']
    }
    
    processed = loader.preprocess(test_samples)
    
    # 验证处理结果
    assert isinstance(processed, dict)
    assert processed['input_ids'].shape == (2, config['dataset']['max_length'])
    assert processed['attention_mask'].shape == (2, config['dataset']['max_length'])

def test_collate_fn(config):
    """测试collate函数"""
    loader = GSM8KDataLoader(config)
    dummy_batch = [
        {'input_ids': torch.tensor([1,2,3]), 'attention_mask': torch.tensor([1,1,1])},
        {'input_ids': torch.tensor([4,5,6]), 'attention_mask': torch.tensor([1,1,1])}
    ]
    
    # 获取collate_fn
    train_loader, _ = loader.load_dataset()
    collated = train_loader.collate_fn(dummy_batch)
    
    # 验证合并结果
    assert collated['input_ids'].shape == (2, 3)
    assert collated['attention_mask'].shape == (2, 3)

if __name__ == "__main__":
    # 命令行测试
    with open('d:/Document/学习/algorithm/rlhf/grpo/configs/config_test.yaml') as f:
        test_config = yaml.safe_load(f)
    
    print("Running smoke tests...")
    test_loader = GSM8KDataLoader(test_config)
    
    # 基本功能测试
    print("Testing tokenizer initialization...")
    assert test_loader.tokenizer is not None
    
    print("Testing dataset loading...")
    train_loader, val_loader = test_loader.load_dataset()
    assert train_loader is not None and val_loader is not None
    
    print("Testing batch shapes...")
    batch = next(iter(train_loader))
    print(f"Batch shapes - input_ids: {batch['input_ids'].shape}, attention_mask: {batch['attention_mask'].shape}")
    
    print("All smoke tests passed!")