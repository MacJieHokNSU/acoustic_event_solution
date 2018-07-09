# Copyrights
#   Author: Alexey Svischev
#   Created: 10/07/2018


import logging

import pickle
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

from src.utils.data_utils import get_labels_to_int_map
from src.utils.data_utils import get_train_pairs_from_ontology
from src.models.vggish_model import VggishModel
from src.features.vggish_input import wavfile_to_examples


LOGGER = logging.getLogger('train onpen log')

model_save_path = 'models/knn_classifier.pkl'
train_data_path = 'data/ontology/mini_audioset'
ontology_map_path = 'data/ontology/classes_map.pkl'
vgg_model_checkpoint = 'models/vggish_model.ckpt'

if __name__ == '__main__':

    LOGGER.info('prepare train pairs')
    samples_pathes, labels = list(
        zip(
            *get_train_pairs_from_ontology(train_data_path)
        )
    )

    LOGGER.info('prepare labels map')
    labels_map = get_labels_to_int_map(labels)
    int_labels = [labels_map[label] for label in labels]

    LOGGER.info('load Vggish model')
    vgg_model = VggishModel(vgg_model_checkpoint)

    LOGGER.info('prepare train features from data')
    features = [
        wavfile_to_examples(sample_path)
        for sample_path in tqdm(samples_pathes, desc='features extracting')
    ]

    LOGGER.info('prepare train vectors')
    train_vectors, train_labels = list(
        zip(
            *[(vgg_model.predict(x), y) for x, y in
                tqdm(zip(features, int_labels), desc='vectors extracting') if len(x) > 0]
        )
    )

    LOGGER.info('train knn classifier')

    classifier = KNeighborsClassifier(n_neighbors=1)
    classifier.fit(train_vectors, train_labels)

    LOGGER.info('save trained model')
    joblib.dump(classifier, model_save_path)

    LOGGER.info('save ontology map')
    with open(ontology_map_path, 'wb') as f:
        pickle.dump(labels_map, f)
