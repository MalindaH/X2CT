# ------------------------------------------------------------------------------
# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Created by Kai Ma (makai0324@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import torchvision
import lpips


class GANLoss(nn.Module):
  def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
    super(GANLoss, self).__init__()
    self.real_label = target_real_label
    self.fake_label = target_fake_label
    self.real_label_tensor = None
    self.fake_label_tensor = None
    if use_lsgan:
      self.loss = nn.MSELoss()
      print('GAN loss: {}'.format('LSGAN'))
    else:
      self.loss = nn.BCELoss()
      print('GAN loss: {}'.format('Normal'))

  def get_target_tensor(self, input, target_is_real):
    target_tensor = None
    if target_is_real:
      create_label = ((self.real_label_tensor is None) or
                      (self.real_label_tensor.numel() != input.numel()))
      if create_label:
        real_tensor = torch.ones(input.size(), dtype=torch.float).fill_(self.real_label)
        self.real_label_tensor = real_tensor.to(input)
      target_tensor = self.real_label_tensor
    else:
      create_label = ((self.fake_label_tensor is None) or
                      (self.fake_label_tensor.numel() != input.numel()))
      if create_label:
        fake_tensor = torch.ones(input.size(), dtype=torch.float).fill_(self.fake_label)
        self.fake_label_tensor = fake_tensor.to(input)
      target_tensor = self.fake_label_tensor
    return target_tensor

  def forward(self, input, target_is_real):
    # for multi_scale_discriminator
    if isinstance(input[0], list):
      loss = 0
      for input_i in input:
        pred = input_i[-1]
        target_tensor = self.get_target_tensor(pred, target_is_real)
        loss += self.loss(pred, target_tensor)
      return loss
    # for patch_discriminator
    else:
      target_tensor = self.get_target_tensor(input[-1], target_is_real)
      return self.loss(input[-1], target_tensor)


class WGANLoss(nn.Module):
  def __init__(self, grad_penalty=False):
    super(WGANLoss, self).__init__()
    self.grad_penalty = grad_penalty
    if grad_penalty:
      print('GAN loss: {}'.format('WGAN-GP'))
    else:
      print('GAN loss: {}'.format('WGAN'))

  def get_mean(self, input):
    input_mean = torch.mean(input)
    return input_mean

  def forward(self, input_fake, input_real=None, is_G=True):
    if is_G:
      assert input_real is None
    cost = 0.
    # for multi_scale_discriminator
    if isinstance(input_fake[0], list):
      for i in range(len(input_fake)):
        if is_G:
          disc_fake = input_fake[i][-1]
          cost += (-self.get_mean(disc_fake))
        else:
          disc_fake = input_fake[i][-1]
          disc_real = input_real[i][-1]
          cost += (self.get_mean(disc_fake) - self.get_mean(disc_real))
      return cost
    # for patch_discriminator
    else:
      if is_G:
        disc_fake = input_fake[-1]
        cost = (-self.get_mean(disc_fake))
      else:
        disc_fake = input_fake[-1]
        disc_real = input_real[-1]
        cost = (self.get_mean(disc_fake) - self.get_mean(disc_real))
      return cost


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        # print(input.device, input.shape, target.shape) # cpu, [4, 3, 128, 128]
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

# Restruction Loss
class RestructionLoss(nn.Module):
  '''
  reduction: 'elementwise_mean' or 'none'
  '''
  def __init__(self, distance='l1', reduction='mean', gpu_ids = []):
    super(RestructionLoss, self).__init__()
    self.distance = distance
    if distance == 'l1':
      self.loss = nn.L1Loss(reduction=reduction)
    elif distance == 'mse':
      self.loss = nn.MSELoss(reduction=reduction)
    elif distance == 'perceptual': # 3d perceptual loss: Euclidean distance between feature vector from pre-trained VGG19
      # self.vgg = torchvision.models.vgg19(pretrained=True)
      # self.vgg.classifier = nn.Sequential(*[self.vgg.classifier[0]])
      # # first_conv_layer = [nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)]
      # # first_conv_layer.extend(list(self.vgg.features))  
      # # self.vgg.features= nn.Sequential(*first_conv_layer)  
      # self.vgg.eval()
      # # print(self.vgg) # classifier one linear layer: 25088 -> 4096
      # # self.loss = None
      self.loss = VGGPerceptualLoss(resize=False)
      if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        print("gpu_ids:",gpu_ids)
        self.loss.to(gpu_ids[0])
        self.loss = torch.nn.DataParallel(self.loss, gpu_ids)
      self.mseloss = nn.MSELoss(reduction=reduction)
    elif distance == 'perceptual_lpips':
      self.loss_fn_vgg = lpips.LPIPS(net='vgg')
    else:
      raise NotImplementedError()
  
  def perceptual_loss(self, gt, pred): # 3d perceptual loss
    assert gt.shape == pred.shape
    # gt = gt.to('cpu')
    # pred = pred.to('cpu')
    i=0
    slice1 = gt[:,:,i,:]
    slice2 = pred[:,:,i,:]
    # slice1 = gt[:,:,i,:].expand(-1, 3, -1,-1)
    # slice2 = pred[:,:,i,:].expand(-1, 3, -1,-1)
    # slice1 = self.vgg(slice1)
    # slice2 = self.vgg(slice2) # 4,4096
    # loss = torch.sqrt(torch.sum((slice1-slice2)**2))
    loss = self.loss(slice1, slice2)
    # print("1",loss)
    for i in range(1,gt.shape[2]):
    # for i in range(gt.shape[2]):
      slice1 = gt[:,:,i,:]
      slice2 = pred[:,:,i,:]
      # slice1 = gt[:,:,i,:].expand(-1, 3, -1,-1)
      # slice2 = pred[:,:,i,:].expand(-1, 3, -1,-1) # [4, 1, 128, 128] -> [4, 3, 128, 128]
      # slice1 = self.vgg(slice1)
      # slice2 = self.vgg(slice2) # [4, 4096]
      # loss += torch.sqrt(torch.sum((slice1-slice2)**2))
      loss += self.loss(slice1, slice2)
    # print("2",loss)
    for i in range(gt.shape[3]):
      slice1 = gt[:,:,:,i,:]
      slice2 = pred[:,:,:,i,:]
      # slice1 = slice1.expand(-1, 3, -1, -1)
      # slice2 = slice2.expand(-1, 3, -1, -1)
      # slice1 = self.vgg(slice1)
      # slice2 = self.vgg(slice2)
      # loss += torch.sqrt(torch.sum((slice1-slice2)**2))
      loss += self.loss(slice1, slice2)
    # print("3",loss)
    for i in range(gt.shape[4]):
      slice1 = gt[:,:,:,:,i]
      slice2 = pred[:,:,:,:,i]
      # slice1 = slice1.expand(-1, 3, -1,-1)
      # slice2 = slice2.expand(-1, 3, -1,-1)
      # slice1 = self.vgg(slice1)
      # slice2 = self.vgg(slice2)
      # loss += torch.sqrt(torch.sum((slice1-slice2)**2))
      loss += self.loss(slice1, slice2)
    loss = torch.sum(loss)
    loss /= gt.shape[2]*gt.shape[3]*gt.shape[4]
    loss *= 100
    print("perceptual loss: ",loss)
    return loss

  def perceptual_loss_lpips(self, gt, pred): # 3d perceptual loss
    assert gt.shape == pred.shape
    # print(torch.min(gt), torch.max(gt))
    # print(torch.min(pred), torch.max(pred))
    gt1 = nn.functional.normalize(gt, dim=2)
    pred1 = nn.functional.normalize(pred, dim=2)
    # print(torch.min(gt1), torch.max(gt1))
    # print(torch.min(pred1), torch.max(pred1))
    i=0
    slice1 = gt1[:,:,i,:].expand(-1, 3, -1,-1)
    slice2 = pred1[:,:,i,:].expand(-1, 3, -1,-1)
    loss = self.loss_fn_vgg(slice1, slice2)
    # slice1 = self.vgg(slice1)
    # slice2 = self.vgg(slice2) # 4,4096
    # loss = torch.sqrt(torch.sum((slice1-slice2)**2))
    for i in range(1,gt.shape[2]):
    # for i in range(gt.shape[2]):
      slice1 = gt1[:,:,i,:].expand(-1, 3, -1,-1)
      slice2 = pred1[:,:,i,:].expand(-1, 3, -1,-1) # [4, 1, 128, 128] -> [4, 3, 128, 128]
      loss += self.loss_fn_vgg(slice1, slice2)
    for i in range(gt.shape[3]):
      slice1 = gt1[:,:,:,i,:]
      slice2 = pred1[:,:,:,i,:]
      slice1 = slice1.expand(-1, 3, -1, -1)
      slice2 = slice2.expand(-1, 3, -1, -1)
      loss += self.loss_fn_vgg(slice1, slice2)
    for i in range(gt.shape[4]):
      slice1 = gt1[:,:,:,:,i]
      slice2 = pred1[:,:,:,:,i]
      slice1 = slice1.expand(-1, 3, -1,-1)
      slice2 = slice2.expand(-1, 3, -1,-1)
      loss += self.loss_fn_vgg(slice1, slice2)
    return loss[0,0,0,0]

  def forward(self, gt, pred):
    print("gt.shape, pred.shape: ",gt.shape, pred.shape) # batch_size,1,128,128,128
    if self.distance == 'perceptual': # 3d perceptual loss
      print("mse loss:",self.mseloss(gt, pred))
      return self.perceptual_loss(gt, pred)
    elif self.distance == 'perceptual_lpips':
      return self.perceptual_loss_lpips(gt, pred)
    else:
      return self.loss(gt, pred)
  


if __name__ == '__main__':
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  print(device)
  # criterionIdt = RestructionLoss('perceptual', 'elementwise_mean').to(device)
  criterionIdt = RestructionLoss('perceptual_lpips', 'elementwise_mean').to(device)
  G_fake_D = torch.rand(1,1,128,128,128).to(device)
  G_real_D = torch.rand(1,1,128,128,128).to(device)
  G_fake_D *= 128
  G_real_D *= 128
  # G_fake_D = torch.tensor([[[ 1,  2,  3],
  #        [ 4,  5,  6]],
  #       [[ 7,  8,  9],
  #        [10, 11, 12]]], dtype=torch.float32)
  # G_real_D = torch.tensor([[[ 1,  2,  3],
  #        [ 4,  5,  6]],
  #       [[ 7,  8,  9],
  #        [10, 11, 12]]], dtype=torch.float32)
  loss_idt = criterionIdt(G_fake_D, G_real_D)
  print(loss_idt)