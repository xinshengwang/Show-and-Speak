import os
import time
import torch
import argparse
import numpy as np
import torchvision.transforms as transforms 
from hparams import hparams as hps
from torch.utils.data import DataLoader
from utils.util import mode
from utils.dataset import I2SData, pad_collate,pad_collate_BU
from model.model import I2SModel, I2SLoss
from waveglow.denoiser import Denoiser
from scipy.io.wavfile import write
import math
import random
import pdb

random_seed = 0 
np.random.seed(random_seed)
random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)   
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):   
	np.random.seed(random_seed + worker_id)


def prepare_dataloaders(fdir,split,args):
    imsize = hps.img_size
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    dataset = I2SData(args,fdir,split,imsize,transform=image_transform)
    collate_fn = pad_collate(hps.n_frames_per_step,args)
    collate_fn_BU = pad_collate_BU(args)
    if split == 'train':
        data_loader = DataLoader(dataset, num_workers = hps.n_workers, shuffle = True,
                                batch_size = hps.batch_size, pin_memory = hps.pin_mem,
                                drop_last = True, collate_fn = collate_fn,worker_init_fn=worker_init_fn)
    else:
        if args.img_format == 'BU':
            data_loader = DataLoader(dataset, num_workers = hps.n_workers, shuffle = False,
                    batch_size = 16, pin_memory = hps.pin_mem,
                    drop_last = False,worker_init_fn=worker_init_fn)  
        else:
            data_loader = DataLoader(dataset, num_workers = hps.n_workers, shuffle = False,
                                batch_size = 16, pin_memory = hps.pin_mem,
                                drop_last = False,worker_init_fn=worker_init_fn)
    return data_loader


def train(model,train_loader,val_loader,args,waveglow=None):
	optimizer = torch.optim.Adam(model.parameters(), lr = hps.lr,
								betas = hps.betas, eps = hps.eps,
								weight_decay = hps.weight_decay)
	criterion = I2SLoss(args)
	model_path = "%s/models" % (args.save_path)
	if not os.path.exists(model_path):
		os.makedirs(model_path)
	epoch = args.start_epoch
	if epoch != 0:
		model.load_state_dict(torch.load("%s/models/I2SModel_%d.pth" % (args.save_path,epoch)))
	
	if hps.sch:
		lr_lambda = lambda step: hps.sch_step**0.5*min((step+1)*hps.sch_step**-1.5, (step+1)**-0.5)
		scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
	
	model.train()
	while epoch <= args.max_epoch:
		iteration = 0
		for batch in train_loader:
			iteration += 1
			start = time.perf_counter()
			x, y = model.parse_batch(batch)
			prob = max(args.k / (args.k + math.exp(epoch/args.k) - 1),(1-args.m))
			ss_prob =1.0 - prob
			y_pred = model(x,ss_prob)
			loss, item = criterion(y_pred, y, epoch)
			model.zero_grad()
			loss.backward()
			grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hps.grad_clip_thresh)
			optimizer.step()
			if hps.sch:
				scheduler.step()

			dur = time.perf_counter()-start
			if iteration % 10 == 0:
				info = 'Epoch: {} Iter: {} Loss: {:.2e} Grad Norm: {:.2e} {:.1f}s/it \n'.format( epoch,
					iteration, item, grad_norm, dur)
				print(info)
				save_file = os.path.join(args.save_path, args.result_file)
				with open(save_file, "a") as file:
					file.write(info)
		if epoch % 25 == 0:			
			infer_mel(model,val_loader,args,epoch)
			model.train()
			torch.save(model.state_dict(),
				"%s/models/I2SModel_%d.pth" % (args.save_path,epoch))			
		epoch += 1

def infer_mel(model,val_loader,args,epoch):
	model.eval()
	for imgs,vis_info, keys in val_loader:
		imgs = imgs.float().cuda()
		vis_info = vis_info.float().cuda()
		with torch.no_grad():
			output = model.inference(imgs,vis_info)
		mel_outputs, mel_outputs_postnet, _ ,_, mel_lengths= output
		
		for j, mel in enumerate(mel_outputs_postnet):
			key = keys[j]
			root = os.path.join(args.save_path,'mels',str(epoch))
			if not os.path.exists(root):
				os.makedirs(root)
			
			path = os.path.join(root,key) + '.npy'
			
			mel = mel[:,:mel_lengths[j]*hps.n_frames_per_step]
			np.save(path,mel.cpu().numpy())
		
def infer(model,val_loader,args,epoch,waveglow=None):
	model.eval()
	i = 0
	for imgs,vis_info, keys in val_loader:
		imgs = imgs.float().cuda()
		vis_info = vis_info.float().cuda()
		with torch.no_grad():
			output = model.inference(imgs,vis_info)
		mel_outputs, mel_outputs_postnet, _ ,_, mel_lengths= output

		with torch.no_grad():
			audios = waveglow.infer(mel_outputs_postnet,sigma=hps.sigma_infer)
			audios = audios.float()
			audios = denoiser(audios, strength=hps.denoising_strength).squeeze(1)
		
		for j, audio in enumerate(audios):
			i += 1
			key = keys[j]
			root = os.path.join(args.save_path,'audios',str(epoch))
			if not os.path.exists(root):
				os.makedirs(root)
			path = os.path.join(root,key) + '.wav'
			audio = audio[:mel_lengths[j]*hps.seft_hop_length*hps.n_frames_per_step]
			audio = audio/torch.max(torch.abs(audio))
			write(path, hps.sample_rate, (audio.cpu().numpy()*32767).astype(np.int16))
			if i % 50 == 0:
				print('processed {} audio'.format(i))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--data_dir', type = str, default = 'Data_for_SAS',
						help = 'directory to load data')
	parser.add_argument('--waveglow_model',type=str,default = 'waveglow_256channels.pt')
	parser.add_argument('-o','--save_path',type=str,default='output')
	parser.add_argument('--img_format',type=str,default='BU',choices=['BU','vector','tensor','img'])
	parser.add_argument('--start_epoch',type=int,default=0)
	parser.add_argument('--max_epoch',type=int,default=1100)
	parser.add_argument('--result_file',type=str,default='results')
	parser.add_argument('--gamma1',type=float,default=0.5,
						help = 'parameter of the image embedding constraint loss')
	parser.add_argument('--only_val',action="store_true",default=False,
						help = 'true for synthesizing speech in inference stage')
	parser.add_argument('--k',type=int,default=160,
						help = 'parameter of the inverse sigmoid in scheduled sampling')
	parser.add_argument('--m',type=float,default=0.025,
						help = 'max sampling rate of inferred spectrogram frames in scheduled sampling')
	parser.add_argument('--scheduled_type',type=str,default='sigmoid',choices=['sigmoid', 'linear','exp'])
	args = parser.parse_args()

	torch.backends.cudnn.enabled = True
	torch.backends.cudnn.benchmark = False # faster due to dynamic input shape
	args.save_path = os.path.join(args.save_path,str(args.k),str(args.m))
	train_loader = prepare_dataloaders(args.data_dir,'train',args)
	val_loader = prepare_dataloaders(args.data_dir,'test',args)	
	model = I2SModel(args)
	mode(model, True)
	if not args.only_val:
		train(model,train_loader,val_loader,args)
	else:	
		waveglow_path = args.waveglow_model
		waveglow = torch.load(waveglow_path)['model']
		denoiser = Denoiser(waveglow).cuda()

		for e in (list(np.arange(400,1101,50))):
			print ('start processing {} epoch'.format(e))
			args.start_epoch = e
			model.load_state_dict(torch.load("%s/models/I2SModel_%d.pth" % (args.save_path,args.start_epoch)))
			infer(model,val_loader,args,args.start_epoch,waveglow)
			