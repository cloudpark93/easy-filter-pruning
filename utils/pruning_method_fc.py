import numpy as np
from kerassurgeon import Surgeon


def pruning_method_fc(model, layer_to_prune, pruning_amount, method):

    if method == 'L1norm':
        # Load surgeon package
        surgeon = Surgeon(model)

        # Store weights from conv layers [0][1] = [weight][bias] Hence, [0] to store weights only
        fc_layer_weights = [model.layers[i].get_weights()[0] for i in layer_to_prune]

        for i in range(len(fc_layer_weights)):
            if pruning_amount[i] == 0:
                continue
            weight = fc_layer_weights[i]
            num_filters = weight.shape[1]
            filter_to_prune = {}

            # compute L1-norm of each filter weight and store it in a dictionary(filter_to_prune)
            for j in range(num_filters):
                L1_norm = np.sum(abs(weight[:, j]))
                filter_number = 'filter_{}'.format(j)
                filter_to_prune[filter_number] = L1_norm

            # sort the filter according to the ascending L1 value
            # pruning[0]: sort by name, pruning[1]: sort by value
            filter_to_prune_sort = sorted(filter_to_prune.items(), key=lambda pruning: pruning[1])
            print(filter_to_prune_sort)

            # extracting filter number from '(filter_2, 0.515..), eg) extracting '2' from '(filter_2, 0.515..)
            remove_channel = [int(filter_to_prune_sort[i][0].split("_")[1]) for i in range(0, pruning_amount[i])]
            print(remove_channel)

            # delete filters with lowest L1 norm values
            surgeon.add_job('delete_channels', model.layers[layer_to_prune[i]], channels=remove_channel)

        model_pruned = surgeon.operate()

        return model_pruned


    
