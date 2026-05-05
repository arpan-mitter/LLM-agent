#3. src/trainer.py
#The main loop. It uses BFloat16 and gradient accumulation to stay fast on your CPU.

#Python

import torch
import time
import os
from model import SLM, SLMConfig
from utils import get_batch

# Config
batch_size = 12
block_size = 128
max_iters = 2000
grad_accum_steps = 4
lr = 5e-4

# Init
config = SLMConfig(block_size=block_size)
model = SLM(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
os.makedirs('checkpoints', exist_ok=True)

print("Starting training on CPU...")
for i in range(max_iters):
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)
    
    for _ in range(grad_accum_steps):
        x, y = get_batch(batch_size, block_size)
        with torch.cpu.amp.autocast(dtype=torch.bfloat16):
            logits, loss = model(x, y)
            loss = loss / grad_accum_steps
        loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    if i % 100 == 0:
        print(f"Iter {i}: Loss {loss.item()*grad_accum_steps:.4f} | Time {time.time()-t0:.2f}s")
        torch.save({'model_state_dict': model.state_dict()}, "checkpoints/last_model.pt")