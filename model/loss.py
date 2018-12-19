from torch import nn


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, data, output):
        logits = output['logits']
        target = data['label']
        return self.loss_fn(logits, target)
