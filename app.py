import base64
import os
import re
from io import BytesIO

import torch

from flask import Flask, render_template, request, make_response
from torchvision import utils
from werkzeug.exceptions import BadRequest

from models.calgen import Generator
from models.edarnet import EDAR
from models.mwcnn import MWCNN
from models.rrdbnet import RRDBNet
from tools.dataloader.load_image import mwcnn_loader, mwcnn_post, gen_loader, gen_post, cal_load, cal_post

app = Flask(__name__)


def model_loader(model, weight_path):
    device = torch.device('cpu')
    mod = model()
    mod.load_state_dict(torch.load(weight_path, map_location=device))
    mod.eval()
    return mod


@app.route('/', defaults={'pagenow': 'index'})
@app.route('/<pagenow>')
def home(pagenow):
    return render_template(pagenow + '.html')


def chck_file(inp_file):
    if not inp_file:
        return BadRequest("File not valid")
    if inp_file.filename == '':
        return BadRequest("File name is not present in request")
    if not inp_file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        return BadRequest("Invalid file type")


@app.before_request
def before_request():
    if re.search(r"/image//*", request.path):
        ik = request.files.get('file')
        br = chck_file(ik)
        if br:
            return br


@app.route('/image/denoise', methods=['POST'])
def denoise():
    inp_file = request.files.get('file')
    pt = request.form.get('strength')
    path = './weights/MWCNN_' + pt + '.pt'
    out_buffer = BytesIO()
    model = model_loader(MWCNN, path)
    load_list = []
    imglist = mwcnn_loader(inp_file)
    for img_ch in imglist:
        with torch.no_grad():
            load_list.append(model(img_ch))
    out = mwcnn_post(load_list)
    out.save(out_buffer, format='PNG')
    out_str = base64.b64encode(out_buffer.getvalue())
    response = make_response(out_str)
    response.headers.set('Content-type', 'image/png')
    return response


@app.route('/image/superresolution', methods=['POST'])
def supres():
    inp_file = request.files.get('file')
    path = './weights/RRDB.pth'
    out_buffer = BytesIO()
    model = model_loader(RRDBNet, path)
    img = gen_loader(inp_file)
    if img.shape[2]*img.shape[3] > 400*400:
        return BadRequest("Image dimensions are larger than expected")
    with torch.no_grad():
        load = model(img)
    out = gen_post(load)
    out.save(out_buffer, format='PNG')
    out_str = base64.b64encode(out_buffer.getvalue())
    response = make_response(out_str)
    response.headers.set('Content-type', 'image/png')
    return response


@app.route('/image/arti', methods=['POST'])
def arti():
    inp_file = request.files.get('file')
    path = './weights/EDAR.pth'
    out_buffer = BytesIO()
    model = model_loader(EDAR, path)
    img = gen_loader(inp_file)
    if img.shape[2]*img.shape[3] > 2000*2000:
        return BadRequest("Image dimensions are larger than expected")
    with torch.no_grad():
        load = model(img)
    out = gen_post(load)
    out.save(out_buffer, format='PNG')
    out_str = base64.b64encode(out_buffer.getvalue())
    response = make_response(out_str)
    response.headers.set('Content-type', 'image/png')
    return response


@app.route('/image/inpaint', methods=['POST'])
def inpaint():
    inp_file = request.files.get('file')
    inp_mask = request.files.get('mask')
    path = './weights/cal.pt'
    out_buffer = BytesIO()
    model = model_loader(Generator, path)
    img, mask = cal_load(inp_file, inp_mask)
    if img.shape[2]*img.shape[3] > 800*800:
        return BadRequest("Image dimensions are larger than expected")
    with torch.no_grad():
        load = model(img, mask)
        load = load * mask + img * (1. - mask)
    utils.save_image(load, out_buffer, normalize=True, padding=0, format='PNG')
    out_str = base64.b64encode(out_buffer.getvalue())
    response = make_response(out_str)
    response.headers.set('Content-type', 'image/png')
    return response


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
