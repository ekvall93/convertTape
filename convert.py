import click
from convertModel import torch2tf

@click.command()
@click.option('--torch_model', type=click.Path(), help="Path to tape torch model.")
@click.option('--lmdb', type=click.Path(), help="Path to LMBD data folder.")
@click.option('--tf_model', type=click.Path(), help="Path to LMBD data folder.")
@click.option('--data_type', default="test", help="Path to LMBD data folder.")
def cli(torch_model, lmdb, tf_model, data_type):
    """Convert pytorch tape model to tensorflow"""
    torch2tf(torch_model, lmdb, tf_model, data_type)
    