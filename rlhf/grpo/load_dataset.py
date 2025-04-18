# load_dataset.py
'''
提取数据集中的prompt和completion
'''
from datasets import load_dataset

class GSM8KDataSet:
    def __init__(self, config):
        self.dataset = load_dataset(
            config['gsm8k']['path'], 
            config['gsm8k']['mode']
        )

    def get_dataset(self, set_type = 'train'):
        datasets = self.dataset[set_type].map(lambda example: {
            'prompt': example['question'],
            'completion': example['answer']
        }).remove_columns(['question', 'answer'])
        return datasets
        
if __name__ == '__main__':
    config = {
        'gsm8k': {
            'path': 'D:\\Document\\学习\\algorithm\\rlhf\\datasets\\openai\\gsm8k',
            'mode':'main'
        }
    }
    gsm_8k_dataset = GSM8KDataSet(config)
    train_dataset = gsm_8k_dataset.get_dataset('train')
    val_dataset = gsm_8k_dataset.get_dataset('test')
    print(train_dataset[0])