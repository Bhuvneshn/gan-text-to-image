import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class discriminator_emb(nn.Module):
	def __init__(self):
		super(discriminator_emb, self).__init__()
		self.emd_size = 1024 
        
		self.netD_1 = nn.Sequential(

			nn.Linear(self.emd_size,512),
            nn.Linear(512,256),
            nn.Linear(256,128),
            nn.Linear(128,64),
            nn.Linear(64,1)
		)

		
	def forward(self, inp):
		output = self.netD_1(inp)
		return output.view(-1, 1).squeeze(1)