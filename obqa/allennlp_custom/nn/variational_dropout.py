"""
This implementation is borrowed https://github.com/rowanz/swagaf/
"""
import torch
from torch.autograd import Variable


class VariationalDropout(torch.nn.Dropout):

    def forward(self, input):
        """
        input is shape (batch_size, timesteps, embedding_dim)
        Samples one mask of size (batch_size, embedding_dim) and applies it to every time step.
        """
        # ones = Variable(torch.ones(input.shape[0], input.shape[-1]))
        ones = Variable(input.data.new(input.shape[0], input.shape[-1]).fill_(1))
        dropout_mask = torch.nn.functional.dropout(ones, self.p, self.training, inplace=False)
        if self.inplace:
            input *= dropout_mask.unsqueeze(1)
            return None
        else:
            return dropout_mask.unsqueeze(1) * input

