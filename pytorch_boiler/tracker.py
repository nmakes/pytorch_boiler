import torch.nn as nn
import numpy as np
import json


class Tracker:

    def __init__(self):
        self.history = {}
        self.running_history = []
    
    def update(self, key, value):
        if key not in self.history:
            self.history[key] = []
        self.history[key].append(value)

    def stash(self):
        self.running_history.append(self.history)
        self.history = {}

    def summarize(self):
        summary = {}
        for key in self.history:
            summary[key] = np.mean(self.history[key]).item()
        return json.dumps(summary, indent=2)

    def state_dict(self):
        state = {
            'history': self.history,
            'running_history': self.running_history
        }
        return state

    def load_state_dict(self, state_dict):
        self.history = state_dict['history']
        self.running_history = state_dict['running_history']
        return self