import os
import csv
import numpy as np
import glob


def conv_pruning_amount_calculator(pruning_method, pruning_index_for_sensitivity, pruning_acc_threshold):

    method = pruning_method
    folder_path = './test_continuous_pruning/{}/conv'.format(method)
    folder_number = os.listdir(folder_path)
    folder_number = len(folder_number)

    file_path = folder_path + '/test_continuous_pruning_layer1'
    file_number = glob.glob(os.path.join(file_path, "*.csv"))
    file_number = len(file_number)

    pruning_index = pruning_index_for_sensitivity
    pruning_acc_threshold = pruning_acc_threshold

    pruned_acc = []

    for layer in range(1, folder_number+1):
        folder = os.path.join(folder_path + '/test_continuous_pruning_layer{}'.format(layer))

        for i in range(1, file_number+1):
            data = csv.DictReader(open(folder+'/vgg16_cifar10_after_prune_{}_{}%.csv'.format(method, pruning_index*i)))
            pruned_acc_for_comparison = []

            for raw in data:
                pruned_acc_for_comparison = list(raw.values())
                pruned_acc_for_comparison = np.array(pruned_acc_for_comparison).astype(np.float32)

            if pruned_acc_for_comparison > pruning_acc_threshold and i == file_number:
                pruned_acc.append(pruning_index * i * 0.01)

            if pruned_acc_for_comparison < pruning_acc_threshold:
                pruned_acc.append(pruning_index * (i - 1) * 0.01)
                break

    print('conv layer pruning amount in each layer: ', pruned_acc)
    return pruned_acc


def fc_pruning_amount_calculator(pruning_method, pruning_index_for_sensitivity, pruning_acc_threshold):

    method = pruning_method
    folder_path = 'test_continuous_pruning/{}/fc'.format(method)
    folder_number = os.listdir(folder_path)
    folder_number = len(folder_number)

    file_path = folder_path + '/test_continuous_pruning_layer1'
    file_number = glob.glob(os.path.join(file_path, "*.csv"))
    file_number = len(file_number)

    pruning_index = pruning_index_for_sensitivity
    pruning_acc_threshold = pruning_acc_threshold

    pruned_acc = []

    for layer in range(1, folder_number+1):
        folder = os.path.join(folder_path + '/test_continuous_pruning_layer{}'.format(layer))

        for i in range(1, file_number+1):
            data = csv.DictReader(open(folder+'/vgg16_cifar10_after_prune_{}_{}%.csv'.format(method, pruning_index*i)))
            pruned_acc_for_comparison = []

            for raw in data:
                pruned_acc_for_comparison = list(raw.values())
                pruned_acc_for_comparison = np.array(pruned_acc_for_comparison).astype(np.float32)

            if pruned_acc_for_comparison > pruning_acc_threshold and i == file_number:
                pruned_acc.append(pruning_index * i * 0.01)

            if pruned_acc_for_comparison < pruning_acc_threshold:
                pruned_acc.append(pruning_index * (i - 1) * 0.01)
                break

    print('fc layer pruning amount in each layer: ', pruned_acc)
    return pruned_acc
