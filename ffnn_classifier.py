from torch import FloatTensor
from torch.nn import Module, Linear
from torch.nn.functional import tanh, log_softmax
from torch.autograd import Variable

class FFNNClassifier(Module):
    def __init__(self, n_inputs, n_hidden, n_outputs):
        super(FFNNClassifier, self).__init__()
        
        self.linear1 = Linear(n_inputs, n_hidden)
        self.linear2 = Linear(n_hidden, n_outputs)
    
    def forward(self, inputs):
        h = tanh(self.linear1(inputs.view(len(inputs), -1)))
        y = self.linear2(h)
        log_probs = log_softmax(y)
        return log_probs

    def predict(self, x):
        log_probs = self.forward(Variable(FloatTensor(x)))
        _, idx = log_probs.data.max(1)
        return idx[0]


