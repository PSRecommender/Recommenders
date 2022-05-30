import sys
import os
import logging
import papermill as pm
import scrapbook as sb
from tempfile import TemporaryDirectory
import numpy as np
import tensorflow.compat.v1 as tf
tf.get_logger().setLevel('ERROR') # only show error messages

from recommenders.utils.timer import Timer
from recommenders.utils.constants import SEED
from recommenders.models.deeprec.deeprec_utils import (
    prepare_hparams
)


from recommenders.models.deeprec.models.sequential.sli_rec import SLI_RECModel as SeqModel
from recommenders.models.deeprec.io.sequential_iterator import SequentialIterator

def set_model():
    ##  ATTENTION: change to the corresponding config file, e.g., caser.yaml for CaserModel, sum.yaml for SUMModel
    # yaml_file = '../../recommenders/models/deeprec/config/sli_rec.yaml'  
    yaml_file = '/tf/Recommenders/sli_rec.yaml'  

    EPOCHS = 10
    BATCH_SIZE = 400
    RANDOM_SEED = SEED  # Set None for non-deterministic result

    data_path = os.path.join("/tf/Recommenders/resources/20220520")
    # for test
    user_vocab = os.path.join(data_path, r'user_vocab.pkl')
    item_vocab = os.path.join(data_path, r'item_vocab.pkl')
    cate_vocab = os.path.join(data_path, r'cate_vocab.pkl')

    train_num_ngs = 4 # number of negative instances with a positive instance for training

    hparams = prepare_hparams(yaml_file, 
                              embed_l2=0., 
                              layer_l2=0., 
                              learning_rate=0.001,  # set to 0.01 if batch normalization is disable
                              epochs=EPOCHS,
                              batch_size=BATCH_SIZE,
                              show_step=20,
                              MODEL_DIR=os.path.join(data_path, "model", "sli_rec/"),
                              SUMMARIES_DIR=os.path.join(data_path, "summary", "sli_rec/"),
                              user_vocab=user_vocab,
                              item_vocab=item_vocab,
                              cate_vocab=cate_vocab,
                              need_sample=True,
                              train_num_ngs=train_num_ngs, # provides the number of negative instances for each positive instance for loss computation.
                )

    input_creator = SequentialIterator


    model_best_trained = SeqModel(hparams, input_creator, seed=RANDOM_SEED)
    path_best_trained = os.path.join(hparams.MODEL_DIR, "best_model")
    print('loading saved model in {0}'.format(path_best_trained))
    model_best_trained.load_model(path_best_trained)
    
    return model_best_trained


def predict(model, userId):
    user_path = os.path.join("/tf/Recommenders/recommend/data/{}".format(userId))

    predict_file = os.path.join(user_path, r'test_{}'.format(userId))
    output_file = os.path.join(user_path, r'output_{}.txt'.format(userId))


    model.predict(predict_file, output_file)


