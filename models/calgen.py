from torch import nn

from tools.cal import CoarseGenerator, FineGenerator


class Generator(nn.Module):
    def __init__(self, use_cuda=False, device_ids=None):
        super(Generator, self).__init__()
        self.input_dim = 3
        self.cnum = 32
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        self.coarse_generator = CoarseGenerator(self.input_dim, self.cnum, self.use_cuda, self.device_ids)
        self.fine_generator = FineGenerator(self.input_dim, self.cnum, self.use_cuda, self.device_ids)

    def forward(self, x, mask):
        x_stage1 = self.coarse_generator(x, mask)
        x_stage2, offset_flow = self.fine_generator(x, x_stage1, mask)
        return x_stage2
