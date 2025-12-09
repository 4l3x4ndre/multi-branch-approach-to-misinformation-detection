from collections import deque

class RunningLoss:
    """
    Multi-mode running loss tracker.
    
    Modes:
    - 'cumulative': classic running average over all updates
    - 'ema': exponential moving average
    - 'sma': simple moving average over a fixed window
    """
    def __init__(self, mode='cumulative', window_size=100, ema_alpha=0.98):
        self.mode = mode
        self.window_size = window_size
        self.ema_alpha = ema_alpha
        self.reset()
    
    def reset(self):
        self.count = 0
        self.sum = 0.0
        self.avg = 0.0
        self.ema = None
        self.window = deque(maxlen=self.window_size)
    
    def update(self, val, n=1):
        if self.mode == 'cumulative':
            self.sum += val * n
            self.count += n
            self.avg = self.sum / (self.count + 1e-12)
            return self.avg
        
        elif self.mode == 'ema':
            if self.ema is None:
                self.ema = val
            else:
                self.ema = self.ema_alpha * self.ema + (1 - self.ema_alpha) * val
            return self.ema
        
        elif self.mode == 'sma':
            for _ in range(n):
                self.window.append(val)
            self.avg = sum(self.window) / (len(self.window) + 1e-12)
            return self.avg
        
        else:
            raise ValueError(f"Unknown mode {self.mode}")
    
    def get(self):
        if self.mode == 'cumulative':
            return self.avg
        elif self.mode == 'ema':
            return self.ema
        elif self.mode == 'sma':
            return self.avg
