from PIL import Image
import torch
import numpy as np
from torchvision import transforms


def mwcnn_loader(inp_img):
    img = Image.open(inp_img).convert('RGB')
    width, height = img.size
    if height >= width > 1024 or height > 1024:
        img = img.resize((1024, 1024))
    img = img.resize((width - (width % 8), height - height % 8))
    imgr, imgg, imgb = img.split()
    imgr = np.array(imgr)
    imgr = imgr[np.newaxis, :]
    imgr = imgr[np.newaxis, :]
    imgr = torch.from_numpy(imgr)
    imgr = imgr.float() / 255.0
    imgg = np.array(imgg)
    imgg = imgg[np.newaxis, :]
    imgg = imgg[np.newaxis, :]
    imgg = torch.from_numpy(imgg)
    imgg = imgg.float() / 255.0
    imgb = np.array(imgb)
    imgb = imgb[np.newaxis, :]
    imgb = imgb[np.newaxis, :]
    imgb = torch.from_numpy(imgb)
    imgb = imgb.float() / 255.0
    return [imgr, imgg, imgb]


def mwcnn_post(imgl):
    imager, imageg, imageb = imgl
    imager = imager.cpu().detach().numpy().squeeze() * 255.0
    imageg = imageg.cpu().detach().numpy().squeeze() * 255.0
    imageb = imageb.cpu().detach().numpy().squeeze() * 255.0
    imager = Image.fromarray(imager.astype(np.uint8))
    imageg = Image.fromarray(imageg.astype(np.uint8))
    imageb = Image.fromarray(imageb.astype(np.uint8))
    rgbimg = Image.merge('RGB', (imager, imageg, imageb))
    return rgbimg


def gen_loader(inp_img):
    img = Image.open(inp_img).convert('RGB')
    img = np.array(img) / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img = img.unsqueeze(0)
    img = img.to(torch.device('cpu'))
    return img


def gen_post(img):
    output = img.squeeze().float().cpu().clamp(0, 1).detach().numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0)) * 255.0
    output = output.round().squeeze()
    output = Image.fromarray(output.astype(np.uint8))
    return output


def cal_load(inp_img, mask):
    img = Image.open(inp_img).convert('RGB')
    mask = Image.open(mask).resize((img.size))
    width, height = img.size
    if width < height:
        img = transforms.CenterCrop((width, width))(img)
        mask = transforms.CenterCrop((width, width))(mask)
    elif height < width:
        img = transforms.CenterCrop((height, height))(img)
        mask = transforms.CenterCrop((height, height))(mask)
    print(img.size, mask.size)
    img = transforms.ToTensor()(img)
    mask = transforms.ToTensor()(mask)[0].unsqueeze(dim=0)
    print(img.shape, mask.shape)
    img = img.mul_(2).add_(-1)
    print(img.shape, mask.shape)
    img = img * (1. - mask)
    img = img.unsqueeze(dim=0)
    mask = mask.unsqueeze(dim=0)
    return [img, mask]


def cal_post(img):
    output = img.squeeze().float().cpu().clamp(0, 1).detach().numpy()
    output = transforms.ToPILImage()(output)
    return output
