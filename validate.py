import click
from tape import ProteinBertForValuePredictionFragmentationProsit
from tape.datasets import PrositFragmentationDatasetHCD
import tensorflow as tf
import numpy as np
import torch
from convertModel import getDataLoader
from tqdm import tqdm
@click.command()
@click.option('--torch_model', type=click.Path(), help="Path to tape torch model.")
@click.option('--lmdb', type=click.Path(), help="Path to LMBD data folder.")
@click.option('--tf_model', type=click.Path(), help="Path to LMBD data folder.")
@click.option('--data_type', default="test", help="Path to LMBD data folder.")
def cli(torch_model, lmdb, tf_model, data_type):
    """Convert pytorch tape model to tensorflow"""

    loader = getDataLoader(lmdb, data_type, batch_size=256)

    pytorch_model = ProteinBertForValuePredictionFragmentationProsit.from_pretrained(torch_model)

    if not tf_model.endswith("/"):
        tf_model += "/"
    #tf_model = tf.keras.models.load_model(f'{tf_model}/model.pb')

    X, Y = list(), list()
    for batch in tqdm(loader):
        keys = ['input_ids', 'collision_energy', 'charge', 'input_mask']
        batch = {k: batch[k] for k in keys}
        tf_batch = {k: batch[k].numpy() for k in keys}

        X.append(pytorch_model(**batch)[0].detach().numpy())
        
        #Y.append(tf_model(**tf_batch))
        


        