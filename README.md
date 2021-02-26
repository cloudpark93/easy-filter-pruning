# easy-filter-pruning
An easy way to conduct filter-pruning for convolutional layers and fully connected layers!

You can use it to determine the sensitivity of individual convolutional & fully connected layer to pruning.  

# co-researcher
Please visit my **co-researcher's** github as well!  
https://github.com/jinsoo9595

# main_training&pruning_sensitivity.py
_Please note that **batch_size** and **learning_rate** highly influence the performance of the trained model._

You can fine-tune the **batch_size, learning_rate,** and **epochs** for training.  
Different values for hyperparatmeters return different performances of models.

The table below shows the history of training an initial model for my case.

|   Model    | Dataset     | Training epochs     | Validation accuracy     |
| :------------- | :----------: | -----------: | -----------: |
|  VGG16 | Cifar10   | 50    | 85%    |
|  VGG16 | Cifar10   | 100   | 84%    |
|  VGG16 | Cifar10   | 150    | 85%    |
|  VGG16 | Cifar10   | 200   | 89%    |
|  VGG16 | Cifar10   | 250    | 90%    |
|  VGG16 | Cifar10   | 300   | 90%    |
|  VGG16 | Cifar10   | 350    | 91%    |
|  VGG16 | Cifar10   | 400   | 91%    |
|  VGG16 | Cifar10   | 450    | 93%    |



This script enables you to prune each convolutional layer by pre-defined value.  
You can use this to determine the sensitivity of individual convolutional layer to pruning.  
  * _We are going to upload the script for pruning fully connected layer soon._
  * _The pruning fc layer function is already implemented in the current script though._


# Models
**Model folder** contains model architectures.  
:alien:*At this moment, only vgg16-cifar10 architecture is included.*

**Original VGG16** model takes an input image of size **224x244**, and gives outpus of **1000 classes**.

**VGG16-cifar10** model takes an input image of size **32x32**, and gives outputs of **10 classes** (depends on what dataset you used).

![vgg16-cifar10 model image](https://user-images.githubusercontent.com/78515689/106845452-4c1ab380-66ee-11eb-970b-e2fdc9b620c2.png)


For [Resnet and Geometric median pruning method](https://github.com/jinsoo9595/interesting-filter-pruning), please take a look at my co-researcher's github!  

# Utils
## 1. Dataset
**Cifar10** and **MNIST** datasets are currently available from the **dataset_loader** script.

We use mean&std to standardize the dataset.  
If you do not want to use the mean&std standardizaton, you can just simply comment out the line.  
*e.g) # x_train, x_test = normalize(x_train, x_test)*

You can also simply uncommnet the below lines to normalize the data to [0, 1] range.  
*# x_train = x_train.astype('float32') / 255.* :point_right: *x_train = x_train.astype('float32') / 255.*  
*# x_test = x_test.astype('float32') / 255.* :point_right: *x_test = x_test.astype('float32') / 255.*

## 2. Pruning job
Our code utilises **keras-surgeon** package by **BenWhetton**.  
Please refer to his github link for details.  
https://github.com/BenWhetton/keras-surgeon  

### 2.1 Pruning convolutional layer
Currently, **L1 norm** is used to determine the importance of the filters.

### 2.2 Pruning fully connected layer
Currently, **L1 norm** is used to determine the importance of the filters.

