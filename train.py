# Copyrights
#   Author: Alexey Svischev
#   Created: 09/07/2018

"""

"""

import logging
import os

import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from tqdm import tqdm

from src.utils.data_utils import get_labels_to_int_map
from src.utils.data_utils import get_train_pairs_from_description
from src.models.vggish_model import VggishModel
from src.features.vggish_input import wavfile_to_examples


LOGGER = logging.getLogger('train log')


data_description_path = 'data/meta/meta.txt'
model_save_path = 'models/svc_classifier.pkl'
vgg_model_checkpoint = 'models/vggish_model.ckpt'

if __name__ == '__main__':

    LOGGER.info('prepare train pairs')
    samples_pathes, labels = list(
        zip(
            *get_train_pairs_from_description(data_description_path)
        )
    )

    LOGGER.info('prepare labels map')
    labels_map = get_labels_to_int_map(labels)
    int_labels = [labels_map[label] for label in labels]

    LOGGER.info('load Vggish model')
    vgg_model = VggishModel(vgg_model_checkpoint)

    LOGGER.info('prepare train features from data')
    train_root = 'data/audio'
    features = [
        wavfile_to_examples(os.path.join(train_root, sample_path))
        for sample_path in tqdm(samples_pathes, desc='features extracting')
    ]

    LOGGER.info('prepare train vectors')
    train_vectors, train_labels = list(
        zip(
            *[(vgg_model.predict(x), y) for x, y in
                tqdm(zip(features, int_labels), desc='vectors extracting') if len(x) > 0]
        )
    )

    LOGGER.info('train SVC classifier')
    classifier = GridSearchCV(
        SVC(
            C=1,
            probability=True,
            class_weight='balanced',
        ),
        param_grid={
            'C': np.arange(20, 40, 1),
            'kernel': [
                'linear',
            ],
        },
        scoring='f1_weighted',
        verbose=False,
        n_jobs=-1
    )

    classifier.fit(train_vectors, train_labels)

    LOGGER.info('save trained model')
    joblib.dump(classifier, model_save_path)
