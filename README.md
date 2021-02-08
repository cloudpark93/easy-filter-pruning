# easy-filter-pruning
An easy way to conduct filter-pruning for Convolutional layers and fully connected layers

# Models
**Model folder** contains model architectures.  
:alien:*At this moment, only vgg16-cifar10 architecture is included.*

**Original VGG16** model takes an input image of size **224x244**, and gives outpus of **1000 classes**.

**VGG16-cifar10** model takes an input image of size **32x32**, and gives outputs of **10 classes** (depends on what dataset you used).

![vgg16-cifar10 model image](https://user-images.githubusercontent.com/78515689/106845452-4c1ab380-66ee-11eb-970b-e2fdc9b620c2.png)


# Utils
## 1. Dataset
**Cifar10** and **MNIST** datasets are currently available from the **dataset_loader** script.

We use mean&std to standardize the dataset.  
If you do not want to use the mean&std standardizaton, you can just simply comment out the line.  
*e.g) # x_train, x_test = normalize(x_train, x_test)*

You can also simply uncommnet the below lines to normalize the data to [0, 1] range.  
*# x_train = x_train.astype('float32') / 255.* :point_right: *x_train = x_train.astype('float32') / 255.*  
*# x_test = x_test.astype('float32') / 255.* :point_right: *x_test = x_test.astype('float32') / 255.*
