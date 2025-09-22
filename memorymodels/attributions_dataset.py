from datasets import load_from_disk, concatenate_datasets
from dotenv import load_dotenv
from tqdm import tqdm
import os


load_dotenv()
checkpoint_root = os.getenv('CHECKPOINT_ROOT')
data_root = os.getenv('DATA_ROOT')

# load attribution dataset and concatenate
attributions_datasets = [
    load_from_disk(f"{data_root}/fineweb-edu-tokenized-train-occlusion-lpad-8k_{i}") for i in range(4)
]
attributions_dataset = concatenate_datasets(attributions_datasets)

# flatten attribution dataset and rename col to match token dataset
attributions_dataset.set_format('torch', columns=['ids', 'memory_attribution'])
id_to_index = {}
for i, batch in tqdm(enumerate(attributions_dataset), total=len(attributions_dataset)):
    for j, id in enumerate(batch['ids']):
        id_to_index[id] = [i, j] # batch_index, position_index
attributions_dataset.rename_column('ids', 'id')

def add_attribution(example):
    id = example['id']
    batch_index, position_index = id_to_index[id]
    attribution = attributions_dataset[batch_index]['memory_attribution'][position_index]
    example['attribution'] = attribution
    return example

# load token dataset, join on id, and save
token_dataset = load_from_disk(f"{data_root}/fineweb-edu-tokenized-train-c1024-lpad-8k")
token_dataset.map(add_attribution)
print(token_dataset[0])
token_dataset.save_to_disk(f"{data_root}/fineweb-edu-tokenized-train-c1024-lpad-attr-8k")
