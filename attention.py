import pizza
from pizza.models import *
import torch as T
import numpy as np


class Seq2SeqAttention(object):

    
    def __init__(self, params):
        self.__dict__.update(params)
        w_init, device = self.w_init, self.device
        n_in, n_out, n_hidden1, n_hidden2 = self.n_in, self.n_out, self.n_hidden1, self.n_hidden2
        
        self.encoder = [LSTMLayer(n_hidden1, n_in), LSTMLayer(n_hidden2, n_hidden1)]
        
        # input-feeding
        self.decoder = [LSTMLayer(n_hidden1, n_in + n_hidden1, trainable_init=False), 
                        LSTMLayer(n_hidden2, n_hidden1, trainable_init=False)]
        
        # global attention using `general` score
        # use a Layer only to train Wa
        self.attention    = DenseLayer(1, 1, softmax_act)
        self.attention.W  = T.ones(1, 1, dtype=self.dtype)
        self.attention.b.requires_grad_(False)  # deactive the bias
        self.attention.Wa = self.attention._w_init(n_hidden2, n_hidden2)[0]
        
        self.output  = [DenseLayer(n_hidden1, n_hidden2 * 2, T.tanh), DenseLayer(n_out, n_hidden1, softmax)]
        self.output[0].hy0 = (0.2*T.rand(n_hidden1, 1, dtype=self.dtype, device=self.device)).requires_grad_()
        
        self.optimizer(self.learning_rate, self.clamping, self.encoder + self.decoder + self.output + [self.attention])
        self.hist_loss = {'train': [], 'val': []}

    
    def forward(self, x, x_context, y, training=True):
        encoder, decoder, attention, output = self.encoder, self.decoder, self.attention, self.output
        
        n_batch  = x.size()[1]
        _y_preds, self.attn_ws = [], []
        
        h1_en, c1_en = encoder[0].forward(x[:, :-1])
        h2_en, c2_en = encoder[1].forward(h1_en)

        h1_de, c1_de  = h1_en[:, -1].view(-1, 1), c1_en[:, -1].view(-1, 1)
        h2_de, c2_de  = h2_en[:, -1].view(-1, 1), c2_en[:, -1].view(-1, 1)
        input_context = x[:, -1].view(-1, 1)
        hy            = output[0].hy0   # trainable init
        for i in range(n_batch):
            x_agg        = T.cat((hy, input_context), dim=0)
            h1_de, c1_de = decoder[0].forward(x_agg, hprev=h1_de, cprev=c1_de)
            h2_de, c2_de = decoder[1].forward(h1_de, hprev=h2_de, cprev=c2_de)
            
            scores       = h2_de.transpose(0, 1).mm(attention.Wa.mm(h2_en)) 
            attn_w       = attention.forward(scores)
            context_v    = (attn_w * h2_en).sum(dim=1).view(-1, 1)
            hy           = output[0].forward(T.cat((context_v, h2_de), dim=0))
            y_pred       = output[1].forward(hy)
            if i < n_batch - 1:
                input_context  = x_context[:, i+1].view(-1, 1) if training else y_pred   # or use argmax 
            _y_preds.append(y_pred); self.attn_ws.append(attn_w.detach())
            
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

