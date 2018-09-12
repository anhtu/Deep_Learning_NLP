import pizza
from pizza.models import *
import torch as T
import numpy as np


class Seq2Seq(object):
    
    def __init__(self, params):
        self.__dict__.update(params)
        w_init, device = self.w_init, self.device
        n_in, n_out, n_hidden1, n_hidden2 = self.n_in, self.n_out, self.n_hidden1, self.n_hidden2
        
        self.encoder = [LSTMLayer(n_hidden1, n_in), LSTMLayer(n_hidden2, n_hidden1)]
        
        self.decoder = [LSTMLayer(n_hidden1, n_in, trainable_init=False), 
                        LSTMLayer(n_hidden2, n_hidden1, trainable_init=False)]
        
        self.output  = DenseLayer(n_out, n_hidden2, softmax)
        
        self.optimizer(self.learning_rate, self.clamping, self.encoder + self.decoder + [self.output])
        self.hist_loss = {'train': [], 'val': []}

    
    def forward(self, x, x_context, y, training=True):
        encoder, decoder = self.encoder, self.decoder
        
        n_batch = x.size()[1]
        h1_en, c1_en = encoder[0].forward(x[:, :-1])
        h2_en, c2_en = encoder[1].forward(h1_en)

        _y_preds = []
        h1_de, c1_de = h1_en[:, -1].view(-1, 1), c1_en[:, -1].view(-1, 1)
        h2_de, c2_de = h2_en[:, -1].view(-1, 1), c2_en[:, -1].view(-1, 1)
        context = x[:, -1].view(-1, 1)
        for i in range(n_batch):
            h1_de, c1_de = decoder[0].forward(context, hprev=h1_de, cprev=c1_de)
            h2_de, c2_de = decoder[1].forward(h1_de,   hprev=h2_de, cprev=c2_de)
            y_pred       = self.output.forward(h2_de)
            if i < n_batch - 1:
                context  = x_context[:, i+1].view(-1, 1) if training else y_pred   # or use argmax 
            _y_preds.append(y_pred)
            
        y_preds = T.cat(_y_preds, dim=1)
        loss    = self.compute_loss(y, y_preds) / n_batch
        
        if training:
            loss.backward()
            self.optimizer.optimize()
                
        return loss, y_preds
              
    
    def train(self, n_iterations, fetcher, hook):
        for j in range(n_iterations+1):
            x, x_context, y, s, inv_s = fetcher()
            loss, y_preds = self.forward(x, x_context, y)
            # logistic
            hook(self, loss, y_preds, j, s, inv_s)
    