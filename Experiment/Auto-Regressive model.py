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
        self.mask = None
    
    def forward(self, x: torch.Tensor):
        if self.mask is None or self.mask.size(0) != len(x):
            # Subsequent mask will mask out tokens from seeing future tokens 
            self.mask = subsequent_mask(len(x)).to(x.device)
            x = self.src_embed(x) # token embeddings with positional encodings
            x = self.encoder(x, self.mask) # transformer encoder
            x = self.generator(x) # get logits 
        return x, None # Second value for state, since trainer is for RNNs also

class Configs(NLPAutoRegressionConfigs):
    model: AutoregressiveTransformer
    transformer: TransformerConfigs

# Transformer configurations
@option(Configs.transformer, 'Transformer')
def _transformer_configs(c: Configs):
    conf = TransformerConfigs()
    conf.n_src_vocab = c.n_tokens
    conf.n_tgt_vocab = c.n_tokens
    conf.d_model = c.d_model
    return conf

# Create GPT model and initialize weights
@option(Configs.model)
def _model(c: Configs):
    m = AutoregressiveTransformer(c.transformer.encoder,
                                  c.transformer.src_embed,
                                  c.transformer.generator).to(c.device)
    
    return m

def main():
    experiment.create(name="transformer")
    conf = Configs()
    experiment.configs(conf, {
      'tokenizer': 'character', # character level tokenizer
      'prompt_separator': '',   
      'prompt': 'It is ',       # starting prompt for sampling
      'text': 'tiny_shakespeare',
      'seq_len': 512,           # context size of 512
      'epochs': 32,             # training for 32 epochs
      'batch_size': 16,         
      'inner_iterations': 10,   # switch between training and validation for 10 times per epoch
      'd_model': 256,
      'transformer.n_heads': 16,
      'transformer.ffn.d_ff': 1024,
      'optimizer.optimizer': 'Noam', # Use Noam optimizer
      'optimizer.learning_rate': 1.,
    })
    experiment.add_pytorch_models({'model': conf.model}) # set models for saving and loading 
    with experiment.start():
        conf.run()

if __name__ == '__main__':
    main()
