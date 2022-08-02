import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class discriminator(nn.Module):
	def __init__(self):
		super(discriminator, self).__init__()
		self.image_size = 64
		self.num_channels = 3
		self.ndf = 64

		self.B_dim = 128
		self.C_dim = 16

		self.netD_1 = nn.Sequential(
			# input is (nc) x 64 x 64
			nn.Conv2d(self.num_channels, self.ndf, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf) x 32 x 32
			nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.ndf * 2),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*2) x 16 x 16
			nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.ndf * 4),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*4) x 8 x 8
			nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.ndf * 8),
			nn.LeakyReLU(0.2, inplace=True)
		)

		self.netD_2 = nn.Sequential(
			# state size. (ndf*8) x 4 x 4
			nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
			nn.Sigmoid()
			)

	def forward(self, inp):
		x_intermediate = self.netD_1(inp)
		output =  self.netD_2(x_intermediate)
		return output.view(-1, 1).squeeze(1), x_intermediate