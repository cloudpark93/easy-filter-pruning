# easy-filter-pruning
An easy way to conduct filter-pruning for convolutional layers and fully connected layers!

You can use it to determine the sensitivity of individual convolutional & fully connected layer to pruning.  

Github repository link for github page visitors: [https://github.com/cloudpark93/easy-filter-pruning](https://github.com/cloudpark93/easy-filter-pruning)

# co-researcher
Please visit my [**co-researcher's**](https://github.com/jinsoo9595) github as well!  
https://github.com/jinsoo9595

# training_initial_model.py
This script is for training an initial model and log the training history in csv files and graph images.

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

# pruning_sensitivity_calculation&plotting.py
This script enables you to prune each convolutional & fully connected layer by a pre-defined value and plots graphs to visualize the results.  

You can use the graphical information to determine the sensitivity of individual convolutional & fully connected layer to pruning.  
![CIFAR10 VGG-16 Pruning Conv Layer Accuracy L1-norm method](https://user-images.githubusercontent.com/78515689/109263569-d31eff80-7846-11eb-9989-2b6d573f1323.jpg)
![CIFAR10 VGG-16 Pruning Fc Layer Accuracy L1-norm method](https://user-images.githubusercontent.com/78515689/109263586-db773a80-7846-11eb-99b9-6be5fa6a4e1f.jpg)


# overall_pruning_wrt_sensitivity.py
This script enables you to prune overall layers (inclusive of both Conv & Fc layers) based on the sensitivitiy analysis, by simply changing the target **pruning accuracy threshold** in the script.  


# Models
**Model folder** contains model architectures.  
:alien:*At this moment, only vgg16-cifar10 architecture is included.*

For [**Resnet and Geometric median pruning method**](https://github.com/jinsoo9595/interesting-filter-pruning), please take a look at my co-researcher's github!

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

## 2. Pruning job
Our code utilises **keras-surgeon** package by **BenWhetton**.  
Please refer to his github link for details.  
https://github.com/BenWhetton/keras-surgeon  

### 2.1 Pruning convolutional layer
Currently, **L1 norm** is used to determine the importance of the filters.

### 2.2 Pruning fully connected layer
Currently, **L1 norm** is used to determine the importance of the filters.

