import torch
from torchsummaryX import summary
from CovidDenseNet import CovidDenseNet
from CovidResNet import CovidResNet


def summarize_network(network, input_size):
    device = torch.device("cpu")
    network = network.to(device)
    summary(network, torch.zeros((1, *input_size)))

covid_densenet = CovidDenseNet(n_classes=2, b_pretrained=True)
summarize_network(covid_densenet, (3,253, 349))

covid_resnet = CovidResNet(n_classes=2, b_pretrained=True)
summarize_network(covid_resnet, (3,253, 349))