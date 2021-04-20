# CovidResNet_CovidDenseNet

This repository contains two deep convolutional neural network (CNN) architectures for automated detection of COVID-19 using computed tomography (CT) images. The first model is called CovidResNet, which is inspired by the deep residual network (ResNet) architecture. The second model is named CovidDenseNet, which exploits the power of densely connected convolutional network (DenseNet) architectures. The networks are designed to provide fast and accurate diagnosis of COVID-19 for multi-class and binary classification tasks.

## CovidResNet:
CovidResNet consists of 29 layers. The first layer is made of 7x7 convolutional filters with a stride of 2, followed by max pooling. We stack four ResNet blocks, each containing bottleneck residual modules. Between every block the spatial dimension is reduced by max pooling and the number of filters is doubled. In the first block, we stack three bottleneck modules, each having three convolutional layers with 64, 64 and 256 filters, respectively. The second block contains three bottleneck modules with a configuration of 128, 128 and 512 filters. In an analogous manner the third block contains 2 modules and the fourth block contains 1 module. The network ends with adaptive average pooling and a fully connected layer. Parts of our proposed architectures exhibit weight configurations that allow initialization with ResNet50 models. All convolutional weights are pre-trained. The weights in the first
convolutional layer and in the first block of CovidResNet are frozen during fine-tuning.

## CovidDenseNet:
The first of the 43 lavers is convolutional with a 7x7 kernel and stride of 2. Then, we apply max pooling and stack four dense blocks interspersed by transition layers. The dense blocks contain 6, 10, 2 and 1 dense layers, respectively. Finally, we perform adaptive average pooling and add a fully connected layer. To ensure the inter-usability of weights between CovidDenseNet and DenseNet121, adapter layers are inserted before transition lavers. They consist of a 1x1 convolution and increase the number of channels to the size required by the subsequent layer. The adapter layers and the last layer are randomly initialized and all other weights are pre-trained. As for CovidResNet, the weights in the first convolutional layer and in the first block of CovidDenseNet are frozen during fine-tuning.

## Dataset Description:

We applied the architectures on the SARS-CoV-2 CT scan dataset, which contains CT images for three different classes: patients with COVID-19, non-COVID-19 viral infections, and healthy individuals. It contains 4173 CT images for 210 subjects structured in a subject-wise manner. The dataset is freely accessible through the link

https://www.kaggle.com/plameneduardo/a-covid-multiclass-dataset-of-ct-scans

## Requirements:

    tensorflow >=1.6
    torchsummaryX
