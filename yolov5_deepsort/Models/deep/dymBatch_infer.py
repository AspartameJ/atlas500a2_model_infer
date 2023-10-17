import sys
import os
import acl

path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path, "acllite"))

from acllite_utils import check_ret
from constants import ACL_MEM_MALLOC_HUGE_FIRST, ACL_MEMCPY_HOST_TO_DEVICE
from acllite_model import AclLiteModel
from acllite_image import AclLiteImage

ITEM_SIZE = 4 # num of bytes


class InferSession(object):
    def __init__(self, model_path, model_width, model_height, max_batch):
        self._model = AclLiteModel(model_path)

        # malloc data memory on device
        self.total_size = max_batch * model_width * model_height * 3 * ITEM_SIZE
        self.total_buffer, ret = acl.rt.malloc(self.total_size, ACL_MEM_MALLOC_HUGE_FIRST)
        check_ret("acl.rt.malloc", ret)

        print("The App arg is __init__")

    def __del__(self):
        if self.total_buffer:
            acl.rt.free(self.total_buffer)  
        print("[Sample] class Samle release source success")

    
    def infer(self, im_batch, batch):
        # load numpy.ndarray to AclLiteImage
        image = AclLiteImage(im_batch)
        # num of bytes of numpy.ndarray
        img_size = image.size
        # copy numpy.ndarray from host to device
        ret = acl.rt.memcpy(self.total_buffer, img_size, image.data(), img_size, ACL_MEMCPY_HOST_TO_DEVICE)
        check_ret("acl.rt.memcpy", ret)
        batch_buffer = {'data': self.total_buffer, 'size':self.total_size}
        return self._model._execute_with_dynamic_batch_size([batch_buffer, ], batch)   


def main():
    #Instance classification detection, pass into the OM model storage path, model input width and height parameters
    MODEL_PATH = "ckpt1_bs1-100.310p3.7.0.rc1.alpha003.om"
    MODEL_WIDTH = 64
    MODEL_HEIGHT = 128
    MAX_BATCH = 100
    infersession = InferSession(MODEL_PATH, MODEL_WIDTH, MODEL_HEIGHT, MAX_BATCH)
    #Read the pictures
    im_batch = ""
    #Reasoning pictures
    batch_size = 11
    result = infersession.infer(im_batch, batch_size)
    print(result[0].shape)

if __name__ == '__main__':
    main()
 
