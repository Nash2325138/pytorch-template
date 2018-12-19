import torch


def top1_acc(data, output):
    with torch.no_grad():
        logits = output['logits']
        target = data['label']
        pred = torch.argmax(logits, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top3_acc(data, output, k=3):
    with torch.no_grad():
        logits = output['logits']
        target = data['label']
        pred = torch.topk(logits, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
