# Copyrights
#   Author: Alexey Svischev
#   Created: 10/07/2018

import logging
import pickle
import os

import numpy as np
from sklearn.externals import joblib
from tqdm import tqdm

from src.utils.data_utils import get_open_pairs
from src.models.vggish_model import VggishModel
from src.features.vggish_input import wavfile_to_examples


model_save_path = 'models/knn_classifier.pkl'
ontology_map_path = 'data/ontology/classes_map.pkl'
result_file_path = 'data/open_task_result.txt'
test_data_path = 'data/test'
vgg_model_checkpoint = 'models/vggish_model.ckpt'

LOGGER = logging.getLogger()


if __name__ == '__main__':

    LOGGER.info('prepare train pairs')
    samples_pathes, labels = list(
        zip(
            *get_open_pairs(test_data_path)
        )
    )

    LOGGER.info('prepare labels map')

    with open(ontology_map_path, 'rb') as f:
        labels_map = pickle.load(f)

    LOGGER.info('load Vggish model')
    vgg_model = VggishModel(vgg_model_checkpoint)

    LOGGER.info('prepare test vectors from data')
    test_root = 'data/test'
    test_vectors = [
        vgg_model.predict(
            wavfile_to_examples(os.path.join(test_root, sample_path))
        ) for sample_path in tqdm(samples_pathes, desc='vectors and features extracting')
    ]

    LOGGER.info('load knn classifier')
    classifier = joblib.load(model_save_path)

    LOGGER.info('predict')
    predict_proba = classifier.predict_proba(test_vectors)
    predict_labels = np.argmax(predict_proba, axis=1)

    inverse_label_map = dict((value, key) for key, value in labels_map.items())

    LOGGER.info('save results')
    with open(result_file_path, 'w') as f:
        for sample_path, probabilities, label in zip(samples_pathes, predict_proba, predict_labels):
            f.write(f'{sample_path} {probabilities[label]} {inverse_label_map[label]} \n')
