# 🐌 slow-transformers

Our motto: "Go transformers! But dont go too fast. You still have to enjoy life ☮️"

> **Diffability**, _noun_
> 
> A principle underscoring the art of unmasking subtle divergences amidst complex similarities, diffability illuminates clear paths through intellectual labyrinths, providing clarity in a sea of cerebral complexity
> ... In practical terms: Understand the difference between two methods by diffing their code them in vscode.

# Install
```
git clone ...
cd slow-transformers/
pip install -r requirements.txt
```

# Supported Models
- [x] ViT
- [ ] SimpleViT
- [x] Language Classification Transformer
- [ ] Encoder-decoder model (generative)

# Supported Datasets
- [x] cifar
- [x] imdb

## TODO / Goals list
- Vanilla transformer (or some language tasks)
- fsdp/deepspeed 
- cross attention
- more interesting architechtures (t5, perciever)
- flash attention integration
- jax?
- resnet & hyena for comparison???
- support m1
- a script to run every model on every possible dataset and record everything in wandb (use hf trainer though)
- also put datasets/dataloading entirely in file (move cifar from ./data to slow_vit.py, similar to hw_vit.py)
