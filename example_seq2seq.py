from .models import Seq2Seq, Seq2SeqAttention 
import pizza as pz
from pz.models import *


# Simple reseresing a string task
characters = 'abcd '  # all str ends with whitespace
int_to_char = list(characters)
char_to_int = {c:i for i, c in enumerate(characters)}

n_vocab = len(characters)

def sample_characters(min_len, max_len):
    """
    Sample a string of characters of random length between `min_len` and `max_len` inclusive
    """
    random_len = np.random.choice(range(min_len, max_len+1))
    char = ''.join(np.random.choice(list(characters[:-1]), size=random_len, replace=True))
    return char, char[::-1]

# Create a dataset of 3000 random string with length ranging from 1 to 15
train_set = [sample_characters(1, 15) for i in range(3000)]
val_set   = [sample_characters(1, 15) for i in range(300)]

def fetcher():
    n_train = len(train_set)
    idx = np.random.choice(range(n_train))
    s, inv_s = train_set[idx]
    s += ' '; inv_s += ' '
    
    x = np.array([one_hot(char_to_int[c]) for c in s]).T
    y = np.array([one_hot(char_to_int[c]) for c in inv_s]).T
    x_context = np.flip(x, axis=1).copy()
    return T.tensor(x, dtype=T.float64, device=running_dev), \
           T.tensor(x_context, dtype=T.float64, device=running_dev), \
           T.tensor(y, dtype=T.float64, device=running_dev), s, inv_s


def hook(model, loss, y_preds, j, s, inv_s):
    print_every = 1000
    
    if j % print_every == 0:
        ix = np.argmax(y_preds.detach().cpu().numpy(), axis=0)
        model.hist_loss['train'].append(loss.detach().cpu().numpy())
        
        print()
        print('[{0}] Loss batch : {1:.4f}'.format(j, loss))
        print('True: {0}'.format(list(inv_s)))
        print('Pred: {0}'.format([int_to_char[c] for c in ix]))
        
        val_loss = 0.
        for i_val in range(len(val_set)):
            s_val, inv_s_val = val_set[i_val]
            s_val += ' '; inv_s_val += ' '
            x = np.array([one_hot(char_to_int[c]) for c in s_val]).T
            y = np.array([one_hot(char_to_int[c]) for c in inv_s_val]).T
        
            _loss, _ = model.forward(T.tensor(x, dtype=T.float64, device=running_dev), None,\
                                     T.tensor(y, dtype=T.float64, device=running_dev), training=False)
            
            val_loss += _loss.detach().cpu().numpy()
        model.hist_loss['val'].append(val_loss)
        print('Valid loss : {1:.4f}'.format(j, val_loss))
        

# Use Seq2Seq model
params = {'n_in': 5, 'n_hidden1': 16, 'n_hidden2': 32, 'n_out': 5,
          'w_init': (None, None), 'device': running_dev,
          'optimizer': pz.Adagrad(), 'learning_rate': 0.1, 'clamping': 50.,
          'compute_loss': bernoulli_likelihood
        }

seq2seq = Seq2Seq(params)
seq2seq.train(n_iterations=10000, fetcher=fetcher, hook=hook) # 30k


# Use global attention model
params = {'n_in': 5, 'n_hidden1': 16, 'n_hidden2': 32, 'n_out': 5, 'dtype': T.float64,
          'w_init': (None, None), 'device': running_dev,
          'optimizer': pz.Adagrad(), 'learning_rate': 0.1, 'clamping': 50.,
          'compute_loss': bernoulli_likelihood
        }

attention_model = Seq2SeqAttention(params)
attention_model.train(n_iterations=20000, fetcher=fetcher, hook=hook) # 30k

