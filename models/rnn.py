import pizza
from pizza.models import *
import torch as T
import numpy as np


class ElmanRNN(object):
    
    def __init__(self, params):
        self.__dict__.update(params)
        n_out, n_embedding, n_hidden, n_context, w_init = self.n_out, self.n_embedding, self.n_hidden, self.n_context, self.w_init
        
        self.embed  = EmbeddingLayer(n_vocab+1, n_embedding, w_init=w_init)  # 1 at the end for padding 
        self.rnn    = RNNLayer(n_embedding * n_context, n_hidden, w_init=w_init)
        self.output = DenseLayer(n_hidden, n_out, activation=softmax, w_init=w_init) 
        
        self.optimizer(self.learning_rate, self.clamping, [self.embed, self.rnn, self.output])
        self.hist_loss = {'train': [], 'val': []}
        
        
    def forward(self, x, y, training=True):
        embed, rnn, output = self.embed, self.rnn, self.output
        
        n_batch = len(x)
        x_embed = T.cat([embed.forward(_x).view(-1, 1) for _x in x], dim=1)
        h       = rnn.forward(x_embed)
        y_preds = output.forward(h)
        
        loss    = self.compute_loss(y, y_preds) / n_batch
        
        if training:
            loss.backward()
            self.optimizer.optimize()
            
            # normalize the embedding
            _embed = embed.embedding.detach()
            embed.embedding = _embed / T.sqrt(_embed.pow(2.).sum(dim=1)).view(-1, 1)
        return loss, y_preds
    
    
    def train(self, n_iterations, fetcher, hook):
        for j in range(n_iterations+1):
            x, y, labels = fetcher()
            loss, y_preds = self.forward(x, y)
            # logistic
            hook(self, loss, y_preds, j, labels) 