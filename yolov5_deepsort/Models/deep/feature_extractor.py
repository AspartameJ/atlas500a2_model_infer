import numpy as np
import cv2
# import aclruntime
# from ais_bench.infer.interface import InferSession
from .dymBatch_infer import InferSession


class Extractor(object):
    def __init__(self, model_path, use_cuda=True):
        self.size = (64, 128)
        self.max_batch = 100
        self.net = InferSession(model_path, self.size[0], self.size[1], self.max_batch) 

    def _preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32)/255., size)
        
        def _normalize(data):
            data = np.transpose(data, (2, 0, 1))
            channel_mean = np.array([0.485, 0.456, 0.406])
            channel_std = np.array([0.485, 0.456, 0.406])
            for i in range(3):
                data[i,:,:] = (data[i,:,:] - channel_mean[i]) / channel_std[i]
  
            return data

        im_batch = np.concatenate([np.expand_dims(_normalize(_resize(im, self.size)), axis=0) for im in im_crops])
        print(im_batch.shape)
        return im_batch


    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        batch_size = im_batch.shape[0]
        features = self.net.infer(im_batch, batch_size)
        return features[0]
    