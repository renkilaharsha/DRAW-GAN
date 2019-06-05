from draw_model import DrawModel
from config import *
#from torchvision.utils import save_image
from utility import *
import torch.nn.utils
import numpy as np
import pandas as pd

torch.set_default_tensor_type('torch.FloatTensor')

model = DrawModel(T,A,B,z_size,N,dec_size,enc_size)

if USE_CUDA:
    model.cuda()

state_dict = torch.load('save/weights_15600.tar')
model.load_state_dict(state_dict)

def save_imag(x,i):
    for t in range(T):
        img = xrecons_grid(x[t],B,A)
        plt.matshow(img, cmap=plt.cm.gray)
        imgname = 'gen_image/i_%d_%s_%d.png' % (i,'gen', t)  # you can merge using imagemagick, i.e. convert -delay 10 -loop 0 *.png mnist.gif
        plt.savefig(imgname)
        print(imgname)
def generate(epoch):
    x,_ = model.generate(batch_size)
    return x
    #save_imag(x,epoch)



global data_ix
if __name__ == '__main__':
    data_ix = np.empty((1,1025))
    for i in range(10):
        x=generate(i)
        save_imag(x,i)
    	x = np.asarray(x[19])
    	lables = np.ones((batch_size,1))
 
    	samples = x.clip(0,1)
    	data_i = np.c_[samples,lables]
            #print(data_i.shape)
    	data_ix = np.r_[data_ix,data_i]
    df = pd.DataFrame(data_ix)
    df.to_csv("gen_draw_telugu_gan_lll.csv",header=None,index=None)
