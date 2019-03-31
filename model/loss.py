from torch import nn


class CrossEntropyLoss(nn.Module):
    def __init__(self, nick_name, weight=1):
        super(CrossEntropyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.nick_name = nick_name
        self.weight = weight

    def forward(self, data_input, model_output):
        logits = model_output['logits']
        target = data_input['label']
        loss = self.loss_fn(logits, target) * self.weight
        return loss
