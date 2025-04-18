# run.py
from load_dataset import GSM8KDataSet
from trl import GRPOConfig, GRPOTrainer
from reward import reward_func, compute_metrics
from peft import LoraConfig


config = {
    'dataset': {
        'gsm8k': {
            'path': 'D:\\Document\\学习\\algorithm\\rlhf\\datasets\\openai\\gsm8k',
            'mode': 'main'
        }
    },
    'GRPOConfig':{
        'output_dir': './Qwen2-0.5B-GRPO',
        'logging_steps': 10,
        'evaluation_strategy': 'steps',  # 按步评估
        'eval_steps': 100,              # 每 100 步评估一次
        'eval_delay': 100,              # 训练 100 步后开始评估
        'deepspeed': 'D:\\Document\\学习\\algorithm\\rlhf\\grpo\\configs\\ds_config.json',
        'log_completions': True,
    },
    'model': 'D:\\Document\\学习\\algorithm\\rlhf\\models\\Qwen\\Qwen2.5-0.5B-Instruct',
    'peft_config': 'D:\\Document\\学习\\algorithm\\rlhf\\grpo\\configs\\peft_config.json',
    
}

gsm_8k_dataset = GSM8KDataSet(config['dataset'])
train_dataset = gsm_8k_dataset.get_dataset('train')
val_dataset = gsm_8k_dataset.get_dataset('test')


training_args = GRPOConfig(**config['GRPOConfig'])

with open(config['peft_config'], "r") as f:
    peft_params = json.load(f)
    peft_config = LoraConfig(**peft_params)

trainer = GRPOTrainer(
    model=config['model'],
    reward_funcs=reward_func,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    peft_config = peft_config,
)
trainer.train()