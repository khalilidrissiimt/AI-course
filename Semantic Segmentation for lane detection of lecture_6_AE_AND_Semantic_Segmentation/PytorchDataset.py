from torch.utils.data import Dataset,DataLoader,random_split
import torch,glob,os
import torchvision.transforms as transforms
from torchvision.io import read_image

class LanesDataset(Dataset):
	def __init__(self,img_dir,label_dir,transforms=None):
		self.img_dir = [i for i in glob.glob(os.path.join(img_dir,r"*\*\*")) if i.split(os.path.sep)[-1]=="20.jpg"]
		self.label_dir = glob.glob(os.path.join(label_dir,r"*\*\*"))
		self.transforms=transforms

	def __len__(self):
		return len(self.img_dir)

	def __getitem__(self,idx):
		image = read_image(self.img_dir[idx])
		label = read_image(self.label_dir[idx])

		if self.transforms:
			image = (self.transforms(image)/255.0)
			label = (self.transforms(label)/255.0)


		return image,label



