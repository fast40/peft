import pathlib
import time
import statistics

from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

SEED = 1337
TEST_HOLDOUT_RATIO = 0.1
DATASET_DIR = pathlib.Path('dataset')
BATCH_SIZE = 1

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

if DATASET_DIR.is_dir() and any(DATASET_DIR.iterdir()):
    dataset = load_from_disk(DATASET_DIR)
else:
    dataset = load_dataset('phanerozoic/Lean4-Mathlib', split='train')
    dataset = dataset.map(lambda x: tokenizer(x['type'] + ' ' + x['fact'], truncation=True), remove_columns=dataset.column_names)
    dataset = dataset.train_test_split(test_size=TEST_HOLDOUT_RATIO, seed=SEED)
    dataset.save_to_disk(DATASET_DIR)


model = AutoModelForCausalLM.from_pretrained('gpt2')

# training loop

collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

num_workers = 8

dataloader = DataLoader(
    dataset['train'],
    batch_size=100,
    collate_fn=collator,
    shuffle=True,
    pin_memory=True,
    num_workers=num_workers,
)


average = []

t = time.perf_counter()
for i, batch in enumerate(dataloader):
    average.append(time.perf_counter() - t)
    t = time.perf_counter()

    if i == 100:
        break;

print(f'Average for {num_workers=}: {statistics.mean(average[1:])}')
plt.title(f'Average: {statistics.mean(average[1:])}')
plt.scatter(range(len(average[1:])), average[1:], marker='.')

for x in range(len(average[1:])):
    plt.axvline(x, color='gray', alpha=0.3, linewidth=0.5)

plt.savefig('output.png', dpi=200)

