import torch


def count(output, target):
    with torch.no_grad():
        predict = torch.argmax(output, 1)
        correct = (predict == target).sum().item()
        return correct
    
    
def save_checkpoint(depth, num_classes, pretrained, epoch, state):
    filename = './checkpoints/checkpoint_' + str(depth)
    filename += '0'*(5-len(str(num_classes))) + str(num_classes)
    if pretrained == True:
        filename += '_T'
    else:
        filename += '_F'
    filename += '_' + '0'*(3-len(str(epoch))) + str(epoch)
    filename += '.pth.tar'
    torch.save(state, filename)