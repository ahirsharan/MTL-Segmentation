
""" Model for meta-transfer learning. """
import  torch
import torch.nn as nn
import torch.nn.functional as F
from models.unet_mtl import UNetMtl
from utils.losses import FocalLoss,CE_DiceLoss,LovaszSoftmax

class BaseLearner(nn.Module):
    """The class for inner loop."""
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.vars = nn.ParameterList()
        self.wt = nn.Parameter(torch.ones([self.args.way,self.args.num_classes,3,3]))
        self.vars.append(self.wt)
        self.bias = nn.Parameter(torch.zeros([self.args.way]))
        self.vars.append(self.bias)
        self.norm1 = nn.BatchNorm2d(self.args.num_classes)
        self.norm2 = nn.ReLU(inplace=True)
        
    def forward(self, input_x, the_vars=None):
        if the_vars is None:
            the_vars = self.vars
        norm1=self.norm1
        norm2=self.norm2
        wt = the_vars[0]
        bias = the_vars[1]
        net=F.conv2d(norm2(norm1(input_x)),wt,bias,stride=1,padding=1)
        return net

    def parameters(self):
        return self.vars

class MtlLearner(nn.Module):
    """The class for outer loop."""
    def __init__(self, args, mode='meta'):
        super().__init__()
        self.args = args
        self.mode = mode
        self.update_lr = args.base_lr
        self.update_step = args.update_step
        self.base_learner = BaseLearner(args)
        num_classes=self.args.num_classes
        if self.mode == 'meta':
            self.encoder = UNetMtl(3,num_classes)  
        else:
            self.encoder = UNetMtl(3,num_classes, mtl=False)  

        self.FL=FocalLoss()
        self.CD=CE_DiceLoss()
        self.LS=LovaszSoftmax()

    def forward(self, inp):
        """The function to forward the model.
        Args:
          inp: input images.
        Returns:
          the outputs of MTL model.
        """
        if self.mode=='train':
            return self.pretrain_forward(inp)
        elif self.mode=='meta':
            data_shot, label_shot, data_query = inp
            return self.meta_forward(data_shot, label_shot, data_query)
        elif self.mode=='val':
            data_shot, label_shot, data_query = inp
            return self.preval_forward(data_shot, label_shot, data_query)
        else:
            raise ValueError('Please set the correct mode.')

    def pretrain_forward(self, inp):
        """The function to forward pretrain phase.
        Args:
          inp: input images(MX3XHXW).
        Returns:
          the outputs of pretrain model(MxCXHXW).
        """
        return self.encoder(inp)

    def meta_forward(self, data_shot, label_shot, data_query):
        """The function to forward meta-train phase.
        Args:
          data_shot: train images for the task
          label_shot: train labels for the task
          data_query: test images for the task.
        Returns:
          logits_q: the predictions for the test samples.
        """
        embedding_query = self.encoder(data_query)
        embedding_shot = self.encoder(data_shot)
        logits = self.base_learner(embedding_shot)
        loss = self.FL(logits, label_shot) + self.CD(logits,label_shot) + self.LS(logits,label_shot)
        grad = torch.autograd.grad(loss, self.base_learner.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.base_learner.parameters())))
        logits_q = self.base_learner(embedding_query, fast_weights)

        for _ in range(1, self.update_step):
            logits = self.base_learner(embedding_shot, fast_weights)
            loss = self.FL(logits, label_shot) + self.CD(logits,label_shot) + self.LS(logits,label_shot)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
            logits_q = self.base_learner(embedding_query, fast_weights)        
        return logits_q

    def preval_forward(self, data_shot, label_shot, data_query):
        """The function to forward meta-validation during pretrain phase.
        Args:
          data_shot: train images for the task
          label_shot: train labels for the task
          data_query: test images for the task.
        Returns:
          logits_q: the predictions for the test samples.
        """
        embedding_query = self.encoder(data_query)
        embedding_shot = self.encoder(data_shot)
        logits = self.base_learner(embedding_shot)
        loss = self.FL(logits, label_shot) + self.CD(logits,label_shot) + self.LS(logits,label_shot)
        grad = torch.autograd.grad(loss, self.base_learner.parameters())
        fast_weights = list(map(lambda p: p[1] - 0.01 * p[0], zip(grad, self.base_learner.parameters())))
        logits_q = self.base_learner(embedding_query, fast_weights)

        for _ in range(1, 100):
            logits = self.base_learner(embedding_shot, fast_weights)
            loss = self.FL(logits, label_shot) + self.CD(logits,label_shot) + self.LS(logits,label_shot)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - 0.01 * p[0], zip(grad, fast_weights)))
            logits_q = self.base_learner(embedding_query, fast_weights)         
        return logits_q
