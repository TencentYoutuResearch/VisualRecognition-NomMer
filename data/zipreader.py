# -*- encoding: utf-8 -*-
# ----------------------------------------------
# filename        :zipreader.py
# description     :NomMer: Nominate Synergistic Context in Vision Transformer for Visual Recognition
# date            :2021/12/28 17:46:06
# author          :clark
# version number  :1.0
# ----------------------------------------------


import zipfile
import io
import numpy as np
from PIL import Image
from PIL import ImageFile
from PIL import UnidentifiedImageError

ImageFile.LOAD_TRUNCATED_IMAGES = True


def is_zip_path(img_or_path):
    """judge if this is a zip path"""
    return '.zip@' in img_or_path


class ZipReader(object):
    """A class to read zipped files"""

    zip_bank = dict()

    def __init__(self):
        super(ZipReader, self).__init__()

    @staticmethod
    def get_zipfile(path):
        zip_bank = ZipReader.zip_bank
        if path not in zip_bank:
            zfile = zipfile.ZipFile(path, 'r')
            zip_bank[path] = zfile
        return zip_bank[path]

    @staticmethod
    def split_zip_style_path(path):
        pos_at = path.index('@')
        assert pos_at != -1, "character '@' is not found from the given path '%s'" % path

        zip_path = path[0:pos_at]
        folder_path = path[pos_at + 1 :]
        folder_path = str.strip(folder_path, '/')
        return zip_path, folder_path

    @staticmethod
    def read(path):
        zip_path, path_img = ZipReader.split_zip_style_path(path)
        zfile = ZipReader.get_zipfile(zip_path)
        data = zfile.read(path_img)
        return data

    @staticmethod
    def imread(path):
        zip_path, path_img = ZipReader.split_zip_style_path(path)
        zfile = ZipReader.get_zipfile(zip_path)
        data = zfile.read(path_img)
        data = data[10:]
        try:
            im = Image.open(io.BytesIO(data))
        except UnidentifiedImageError:
            print("ERROR IMG LOADED: ", path_img)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))

        return im