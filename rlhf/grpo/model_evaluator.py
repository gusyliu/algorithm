from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_loader import GSM8KDataLoader

class ModelEvaluator:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化数据加载器
        self.data_loader = GSM8KDataLoader(config)
        _, self.eval_loader = self.data_loader.load_dataset()
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
        
    def load_model(self, model_path):
        """加载指定路径的模型"""
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model.to(self.device)
        model.eval()
        return model

    def evaluate_model(self, model):
        """评估单个模型"""
        correct = 0
        total = 0
        
        print("\nEvaluation Loader Info:")
        print(f"Dataset length: {len(self.eval_loader.dataset)}")
        print(f"Batch size: {self.eval_loader.batch_size}")
        print(f"Number of batches: {len(self.eval_loader)}")
        print(f"Sampler type: {type(self.eval_loader.sampler).__name__}")
        
        with torch.no_grad():
            for batch in tqdm(self.eval_loader, desc="Evaluating"):
                # print(type(batch['input_ids']))
                # print(batch['input_ids'][0].shape)

                inputs = {
                    'input_ids': torch.tensor(batch['input_ids'], dtype=torch.long).to(self.device),
                    'attention_mask': torch.tensor(batch['attention_mask'], dtype=torch.long).to(self.device)
                }
                
                outputs = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=self.config['eval']['max_length'],
                    num_beams=self.config['eval']['num_beams']
                )
                
                # 解码预测结果
                preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                
                # 这里添加您的评估逻辑
                # 示例: 比较预测结果和真实答案
                correct += sum(1 for pred, ans in zip(preds, batch['answers']) if pred == ans)
                total += len(batch['answers'])
        
        return correct / total if total > 0 else 0

    def compare_models(self, old_model_path, new_model_path):
        """比较新旧模型表现"""
        print("Loading models...")
        old_model = self.load_model(old_model_path)
        new_model = self.load_model(new_model_path)
        
        print("Evaluating old model...")
        old_score = self.evaluate_model(old_model)
        
        print("Evaluating new model...")
        new_score = self.evaluate_model(new_model)
        
        print(f"\nEvaluation Results:")
        print(f"Old Model ({old_model_path}): {old_score:.2%}")
        print(f"New Model ({new_model_path}): {new_score:.2%}")
        print(f"Improvement: {new_score - old_score:.2%}")

# 示例配置
default_config = {
    'model': {
        'name': 'D:/Document/学习/algorithm/rlhf/models/Qwen/Qwen2.5-0.5B'
    },
    'dataset': {
        'gsm8k_dir': 'D:/Document/学习/algorithm/rlhf/datasets/openai/gsm8k',
        'batch_size': 8,
        'max_length': 512,
        'num_workers': 2
    },
    'eval': {
        'max_length': 128,
        'num_beams': 5
    }
}

if __name__ == "__main__":
    evaluator = ModelEvaluator(default_config)
    model = evaluator.load_model(default_config['model']['name'])
    evaluator.evaluate_model(model)
    # evaluator.compare_models(
    #     old_model_path="path/to/old/model",
    #     new_model_path="path/to/new/model"
    # )