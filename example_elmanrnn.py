from .models import ElmanRNN
import pizza as pz
from pz.models import *

# The dataset is ATIS (Airline Travel Information System) dataset
DATAPATH = ''  # 

def atisfold(fold):
    assert fold in range(5)
    train_set, valid_set, test_set, dicts = cPickle.load(gzip.open(DATA_PATH + 'atis.fold' + str(fold) + '.pkl.gz', 'rb'), encoding='latin1')
    return train_set, valid_set, test_set, dicts

train, val, test, dic = atisfold(1)

words_to_idx, ne_to_idx, labels_to_idx = dic['words2idx'], dic['tables2idx'], dic['labels2idx']

idx_to_words  = {v:k for k,v in words_to_idx.items()}
idx_to_ne     = {v:k for k,v in ne_to_idx.items()}
idx_to_labels = {v:k for k,v in labels_to_idx.items()}

test_x,  test_ne,  test_label  = test
val_x,   val_ne,   val_label   = val
train_x, train_ne, train_label = train



def contextwin(l, win):
    """
    win :: int corresponding to the size of the window
    given a list of indexes composing a sentence

    l :: array containing the word indexes

    it will return a list of list of indexes corresponding
    to context windows surrounding each word in the sentence
    """
    assert (win % 2) == 1
    assert win >= 1
    l = list(l)

    lpadded = win // 2 * [-1] + l + win // 2 * [-1]
    out = [lpadded[i:(i + win)] for i in range(len(l))]

    assert len(out) == len(l)
    return out


n_context = 7
running_dev = T.device('cpu')

def fetcher():
    n_train = len(train_x)
    idx = np.random.choice(range(n_train))
    x, y = contextwin(train_x[idx], n_context), train_label[idx]
    _y = np.array([one_hot(_y, n_labels) for _y in y]).T
    return x, T.tensor(_y, dtype=T.float64, device=running_dev), y


def hook(model, loss, y_preds, j, labels):
    print_every = 1000
    
    if j % print_every == 0:
        ix = np.argmax(y_preds.detach().cpu().numpy(), axis=0)
        model.hist_loss['train'].append(loss.detach().cpu().numpy())
        
        print()
        print('[{0}] Loss batch : {1:.4f}'.format(j, loss))
        print('True: {0}'.format(labels))
        print('Pred: {0}'.format(ix))
        
        # Full validation loss
        n_val    = len(val_x)
        val_loss = 0.
        for idx_val in range(n_val):
            x_val, y_val = contextwin(val_x[idx_val], n_context), val_label[idx_val]
            _y_val = T.tensor(np.array([one_hot(_y, n_labels) for _y in y_val]).T, dtype=T.float64, device=running_dev)
            _val_loss, _ = model.forward(x_val, _y_val, training=False)
            val_loss += _val_loss.detach().cpu().numpy()
        val_loss /= n_val
        
        model.hist_loss['val'].append(val_loss)
        print('Val Loss : {1:.4f}'.format(j, val_loss))
        


params = {'n_vocab': n_vocab, 'n_embedding': 100, 'n_out': n_labels, 'n_hidden': 100, 'n_context': n_context,
          'w_init': ('uniform', 0.2), 
          'optimizer': pz.Adagrad(), 'learning_rate': 0.1, 'clamping': 20.,
          'compute_loss': bernoulli_likelihood
         }

rnn_model = ElmanRNNModel(params)
rnn_model.train(n_iterations=50000, fetcher=fetcher, hook=hook) 


