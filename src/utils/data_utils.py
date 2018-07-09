# Copyrights
#   Author: Alexey Svischev
#   Created: 09/07/2018

"""

"""

import os
from typing import Dict
from typing import List
from typing import Tuple


def get_train_pairs_from_ontology(data_path: str) -> List[Tuple[str, str]]:
    """

    :param data_path:
    :return:
    """
    result = []
    for class_dirr in os.listdir(data_path):
        full_class_path = os.path.join(data_path, class_dirr)
        for filename in os.listdir(full_class_path):
            full_sample_path = os.path.join(full_class_path, filename)
            result.append((full_sample_path, class_dirr))
    return result

def get_train_pairs_from_description(data_description_path: str) -> List[Tuple[str, str]]:
    """

    :param data_description_path:
    :return:
    """
    result = []
    with open(data_description_path, 'r') as f:
        for line in f:
            result.append((line.split()[0], line.split()[-1]))
    return result

def get_labels_to_int_map(labels: str) -> Dict[str, int]:
    """

    :param labels:
    :return:
    """
    return dict(zip(sorted(list(set(labels))), range(len(labels))))

def get_test_pairs(test_data_path: str) -> List[Tuple[str, str]]:
    """

    :param test_data_path:
    :return:
    """
    test_samples_pathes = os.listdir(test_data_path)
    test_samples_labels = [x.split('_')[0] for x in test_samples_pathes]
    return [(x, y) for x, y in zip(test_samples_pathes, test_samples_labels) if y != 'unknown']

def get_open_pairs(test_data_path: str) -> List[Tuple[str, str]]:
    """

    :param test_data_path:
    :return:
    """
    test_samples_pathes = os.listdir(test_data_path)
    test_samples_labels = [x.split('_')[0] for x in test_samples_pathes]
    return [(x, y) for x, y in zip(test_samples_pathes, test_samples_labels) if y == 'unknown']
