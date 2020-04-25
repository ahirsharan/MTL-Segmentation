
""" Tools for GPU. """
import os

def set_gpu(cuda_device):
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device
    print('Using gpu:', cuda_device)
    