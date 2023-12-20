# This trains a simple transformer on an NLP auto-regression task(with Tiny Shakespeare dataset)
# https://nn.labml.ai/transformers/basic/autoregressive_experiment.html
import torch
from torch import nn

from labml import experiment
from labml.configs import option
from labml_nn.experiments.nlp_autoregression import NLPAutoRegressionConfigs
from labml_nn.transformers import TransformerConfigs, Encoder
from labml_nn.transformers.utils import subsequent_mask

class AutoregressiveTransformer(nn.Module):
    def __init__(self, encoder: Encoder, src_embed: nn.Module, generator: nn.Module):
        super().__init__()
        self.src_embed = src_embed
        self.encoder = encoder
        self.generator = generator

        