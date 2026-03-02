import pathlib

from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from torch import optim

SEED = 1337
TEST_HOLDOUT_RATIO = 0.1
DATASET_DIR = pathlib.Path('dataset')
BATCH_SIZE = 16

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

if DATASET_DIR.is_dir() and any(DATASET_DIR.iterdir()):
    dataset = load_from_disk(DATASET_DIR)
else:
    dataset = load_dataset('phanerozoic/Lean4-Mathlib', split='train')
    dataset = dataset.map(lambda x: tokenizer(x['type'] + ' ' + x['fact'], truncation=True), remove_columns=dataset.column_names)
    dataset = dataset.train_test_split(test_size=TEST_HOLDOUT_RATIO, seed=SEED)
    dataset.save_to_disk(DATASET_DIR)


collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

dataloader = DataLoader(
    dataset['train'],
    batch_size=BATCH_SIZE,
    collate_fn=collator,
    shuffle=True,
    pin_memory=True,
    num_workers=4,
)

model = AutoModelForCausalLM.from_pretrained('gpt2')

optimizer = optim.AdamW(model.parameters(), lr=3e-4)

for batch in dataloader:
    outputs = model(**batch)
    
    optimizer.zero_grad()
    outputs.loss.backward()
    optimizer.step()

    print(outputs.loss)
