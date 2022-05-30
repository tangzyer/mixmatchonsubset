import numpy as np
import random
import math


# turn label to subset

class subset:

    def __init__(self, num_of_total_classes):
        self.total_classes = num_of_total_classes
        self.universe_set = set(range(0, self.total_classes))
        pai = [pow(2, i)for i in range(2, self.total_classes - 1)]
        dist_of_subset_size = pai / np.linalg.norm(pai, 1)
        self.dist_of_subset_size = dist_of_subset_size

    def index_to_subset(self, class_index):
        size_of_subset = np.random.choice(range(2, self.total_classes - 1), p=self.dist_of_subset_size)
        orig_subset = random.sample(list(self.universe_set), size_of_subset)
        if class_index in orig_subset:
            return orig_subset
        else:
            return list(self.universe_set - set(orig_subset))

    def one_hot_to_obfuscated(self, one_hot_array):
        index = one_hot_array.argmax()
        subset = self.index_to_subset(index)
        ob_array = [0] * self.total_classes
        for i in subset:
            ob_array[i] = 1/len(subset)
        return ob_array

    def index_to_obfuscated(self, index):
        subset = self.index_to_subset(index)
        ob_array = [0] * self.total_classes
        for i in subset:
            ob_array[i] = 1 / len(subset)
        return len(subset), subset, ob_array
    
    def index_to_stack_obfuscated(self, index):
        subset = self.index_to_subset(index)
        stack_y = []
        for i in subset:
          stack_y.append(self.index_to_onehot(i))
        return len(subset), subset, stack_y


    def distill(self, model_outputs, labels, max_size):
        new_label_matrix = np.zeros(shape = model_outputs.shape)
        #size_of_label = np.countnonzero(label)
        #index_of_subset = np.nonzero(label)
        for index, label in labels:
            index_of_subset = np.nonzero(label)[0]
            size_of_subset = index_of_subset.shape[0]
            if size_of_subset == max_size:
                model_output = model_outputs[index]
                a=model_output[index_of_subset].argsort()

    def index_to_onehot(self, index):
        label = [0]*self.total_classes
        label[index] = 1
        return label

    def index_to_subset_multi_hot(self, class_index):
        size_of_subset = np.random.choice(range(2, self.total_classes - 1), p=self.dist_of_subset_size)
        orig_subset = random.sample(list(self.universe_set), size_of_subset)
        if class_index in orig_subset:
            multi_hot = np.zeros(self.total_classes)
            multi_hot[orig_subset] = 1
        else:
            multi_hot = np.ones(self.total_classes)
            multi_hot[orig_subset] = 0
        return multi_hot
    
    def index_to_limited_subset(self, class_index, subset_size=2):
        union = set(range(0, self.total_classes))
        union.remove(class_index)
        subset = random.sample(list(union), subset_size-1)
        subset.append(class_index)
        #orig_subset = random.sample(list(self.universe_set), size_of_subset)
        multi_hot = np.zeros(self.total_classes)
        multi_hot[subset] = 1
        return multi_hot
        