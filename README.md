# GPT2 Zinc 87m

This is a GPT2 style autoregressive language model trained on ~480m SMILES strings from the [ZINC database](https://zinc.docking.org/) available through [Huggingface](https://huggingface.co/entropy/gpt2_zinc_87m).

The model has ~87m parameters and was trained for 175000 iterations with a batch size of 3072 to a validation loss of ~.615. This model is useful for generating druglike molecules or generating embeddings from SMILES strings

## How to use
To use, install the [transformers](https://github.com/huggingface/transformers) library:

```
pip install transformers
```

Load the model from the Huggingface Hub:

```python
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

tokenizer = GPT2TokenizerFast.from_pretrained("entropy/gpt2_zinc_87m", max_len=256)
model = GPT2LMHeadModel.from_pretrained('entropy/gpt2_zinc_87m')
```

To generate molecules:

```python
inputs = torch.tensor([[tokenizer.bos_token_id]])

gen = model.generate(
              inputs,
              do_sample=True, 
              max_length=256, 
              temperature=1.,
              early_stopping=True,
              pad_token_id=tokenizer.pad_token_id,
              num_return_sequences=32
                         )
smiles = tokenizer.batch_decode(gen, skip_special_tokens=True)
```

To compute embeddings:

```python
from transformers import DataCollatorWithPadding

collator = DataCollatorWithPadding(tokenizer, padding=True, return_tensors='pt')

inputs = collator(tokenizer(smiles))
outputs = model(**inputs, output_hidden_states=True)
full_embeddings = outputs[-1][-1]
mask = inputs['attention_mask']
embeddings = ((full_embeddings * mask.unsqueeze(-1)).sum(1) / mask.sum(-1).unsqueeze(-1))
```

## Model Performance

To test generation performance, 1m compounds were generated at various temperature values. Generated compounds were checked for uniqueness and structural validity.

* `percent_unique` denotes `n_unique_smiles/n_total_smiles`
* `percent_valid` denotes `n_valid_smiles/n_unique_smiles`
* `percent_unique_and_valid` denotes `n_valid_smiles/n_total_smiles`

|    |   temperature |   percent_unique |   percent_valid |   percent_unique_and_valid |
|---:|--------------:|-----------------:|----------------:|---------------------------:|
|  0 |          0.5  |         0.928074 |        1        |                   0.928074 |
|  1 |          0.75 |         0.998468 |        0.999967 |                   0.998436 |
|  2 |          1    |         0.999659 |        0.999164 |                   0.998823 |
|  3 |          1.25 |         0.999514 |        0.99351  |                   0.993027 |
|  4 |          1.5  |         0.998749 |        0.970223 |                   0.96901  |


