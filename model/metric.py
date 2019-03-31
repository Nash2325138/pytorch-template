import torch


class TopKAcc():
    def __init__(self, k):
        self.k = k
        self.__name__ = f'top{self.k}_acc'

    def __call__(self, data, output):
        with torch.no_grad():
            logits = output['logits']
            target = data['label']
            pred = torch.topk(logits, self.k, dim=1)[1]
            assert pred.shape[0] == len(target)
            correct = 0
            for i in range(self.k):
                correct += torch.sum(pred[:, i] == target).item()
        return correct / len(target)
