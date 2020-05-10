from utils.common_utils import *
import cv2
import torch.nn.functional as F
import torchvision.transforms as transforms
from skimage.io import imsave
import scipy.io as scio

dtype = torch.FloatTensor

transform = transforms.Compose([
    transforms.ToTensor(),
    ]
)
for i in range(10,30):
	_, raw = get_image('testfile/test/holo/{}.bmp'.format(i))
	A = scio.loadmat('kernel.mat')
	A = A['kernel']
	A = A.reshape(1,A.shape[0],A.shape[1])
	A = torch.from_numpy(A)
	A = A.view(-1,A.shape[1],A.shape[2])
	At = torch.from_numpy(transpose_kernel(A).copy())
	A = A.type(dtype)
	At = At.type(dtype)
	raw = torch.from_numpy(np.array([raw])).type(dtype)
	print(raw)
	model = torch.load('testfile/net0411.pth', map_location='cpu')
	res = model(raw,A,At).detach().numpy().squeeze(0)
	print(res)
	# # tmp = model(raw)
	# # tmp = tmp.detach().numpy().squeeze(0)
	res = np_to_pil(res)
	res.save('testfile/out/res{}.png'.format(i))
	# #res=decov('holo1.bmp','holo1_k.bmp')
	# #res = cv2.bilateralFilter(res,9,75,75)
	# res  = torch_to_np(res).squeeze()
	#imsave('res.png',res)

