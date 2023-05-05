
import torch.optim.lr_scheduler as lr_scheduler

class MyScheduler:
    def __init__(self, optimizer, num_step, epochs, warmup=True, warmup_epochs=1, warmup_factor=1e-3):
        assert num_step > 0 and epochs > 0
        if warmup is False:
            warmup_epochs = 0

        self.optimizer = optimizer
        self.num_step = num_step
        self.epochs = epochs
        self.warmup = warmup
        self.warmup_epochs = warmup_epochs
        self.warmup_factor = warmup_factor

    def schedule(self, x):
        if self.warmup and x <= (self.warmup_epochs * self.num_step):
            alpha = float(x) / (self.warmup_epochs * self.num_step)
            return self.warmup_factor * (1 - alpha) + alpha
        else:
            return (1 - (x - self.warmup_epochs * self.num_step) / ((self.epochs - self.warmup_epochs) * self.num_step)) ** 0.9

    def __call__(self, epoch):
        self.optimizer.param_groups[0]['lr'] = self.schedule(epoch)
