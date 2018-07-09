# Copyrights
#   Author: Alexey Svischev
#   Created: 09/07/2018

import logging
import os

import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from src.utils.data_utils import get_labels_to_int_map
from src.utils.data_utils import get_test_pairs
from src.models.vggish_model import VggishModel
from src.features.vggish_input import wavfile_to_examples


test_data_path = 'data/test'
model_save_path = 'models/svc_classifier.pkl'
result_file_path = 'data/result.txt'
vgg_model_checkpoint = 'models/vggish_model.ckpt'

LOGGER = logging.getLogger()


if __name__ == '__main__':

    LOGGER.info('prepare test pairs')
    samples_pathes, labels = list(
        zip(
            *get_test_pairs(test_data_path)
        )
    )

    LOGGER.info('prepare labels map')
    labels_map = get_labels_to_int_map(labels)

    int_labels = [labels_map[label] for label in labels]

    LOGGER.info('load Vggish model')
    vgg_model = VggishModel(vgg_model_checkpoint)

    LOGGER.info('prepare test vectors from data')
    test_root = 'data/test'
    test_vectors = [
        vgg_model.predict(
            wavfile_to_examples(os.path.join(test_root, sample_path))
        ) for sample_path in tqdm(samples_pathes, desc='vectors and features extracting')
    ]

    LOGGER.info('load classifier')
    classifier = joblib.load(model_save_path)

    LOGGER.info('predict')
    predict_proba = classifier.predict_proba(test_vectors)
    predict_labels = np.argmax(predict_proba, axis=1)

    print(f'accuracy: {accuracy_score(int_labels, predict_labels)}')

    inverse_label_map = dict((value, key) for key, value in labels_map.items())

    LOGGER.info('save results')
    with open(result_file_path, 'w') as f:
        for sample_path, probabilities, label in zip(samples_pathes, predict_proba, predict_labels):
            f.write(f'{sample_path} {probabilities[label]} {inverse_label_map[label]} \n')
