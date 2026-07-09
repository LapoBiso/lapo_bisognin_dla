from collections import deque
import random
import numpy as np
import torch

class replayMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen = capacity)
    
    def append(self, old_state, action, reward, new_state, done):
        state = np.array(old_state, dtype = np.uint8)
        next_state = np.array(new_state, dtype = np.uint8)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch =  random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        pin = torch.cuda.is_available()
        tensors = (
            torch.as_tensor(np.array(states), dtype=torch.uint8),
            torch.as_tensor(actions, dtype=torch.int64),
            torch.as_tensor(rewards, dtype=torch.float32),
            torch.as_tensor(np.array(next_states), dtype=torch.uint8),
            torch.as_tensor(dones, dtype=torch.float32),
        )
        return tuple(t.pin_memory() for t in tensors) if pin else tensors
    
    def __len__(self):
        return len(self.buffer)

def set_seed(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

def load_model(model, path):
    return model.load_state_dict(torch.load(path))
