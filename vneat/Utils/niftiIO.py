import nibabel as nib
from nibabel.freesurfer import io, mghformat
import SimpleITK as sitk
from os.path import join, basename, dirname
import numpy as np
import csv

niiFile = nib.Nifti1Image
mghFile = mghformat.MGHImage


def get_atlas_image_labels(results_io, atlas_path, atlas_dict_path):
    aal_labels_dict = {}

    atlas = results_io.loader(atlas_path).get_data()
    with open(atlas_dict_path, newline='') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            aal_labels_dict[int(row['ROIid'])] = row['ROIname']

    return atlas, aal_labels_dict

def file_reader_from_extension(extension):

    if '.nii' in extension or '.nii.gz' in extension:
        return NiftiReader
    elif '.mgz' in extension or '.mgh' in extension:
        return mghReader
    else:
        return SurfaceReader

def file_writer_from_extension(extension):

    if '.nii' in extension or '.nii.gz' in extension:
        return NiftiWriter
    elif '.mgz' in extension or '.mgh' in extension:
        raise ValueError('MGZ reader not implemented, yet')
    else:
        return SurfaceWriter

def read_surface(filename):

    extension = filename.split('.')[-1]

    if extension == 'mha':
        img = sitk.ReadImage(filename)
        return sitk.GetArrayFromImage(img)
    elif extension == 'annot':
        raise ValueError('Reader for extensions \'label\' not yet implemented')

    elif extension == 'label':
        raise ValueError('Reader for extensions \'inflated\', \'pial\', \'white\' not yet implemented')

    elif extension in ['inflated', 'pial', 'white']:
        coords, faces = io.read_geometry(filename)
        return coords, faces

    else:
        return io.read_morph_data(filename)




class NiftiReader:
    def __init__(self, filename, x1=0, y1=0, z1=0, x2=None, y2=None, z2=None):
        self.filename = filename
        f = nib.load(filename)
        self.niiImage = f
        self.dims = f.shape
        del f
        if x2 == None:
            x2 = self.dims[0]
        if y2 == None:
            y2 = self.dims[1]
        if z2 == None:
            z2 = self.dims[2]

        assert x1 < x2
        assert y1 < y2
        assert z1 < z2

        self.mem_usage = 0.5

        x1, y1, z1 = map(lambda a: max(a, 0), [x1, y1, z1])
        x2, y2, z2 = map(min, zip(self.dims, [x2, y2, z2]))

        self.dims = (x2 - x1, y2 - y1, z2 - z1) + self.dims[3:]
        self.coords = (x1, y1, z1)

    def chunks(self, mem_usage=None):
        if mem_usage != None:
            self.mem_usage = mem_usage

        d = 1.0
        for x in self.dims[3:]:
            d *= x
        nelems = self.mem_usage * (2 ** 17) / d  # (number of MBytes x 2**20 bytes/MB x 8 bits/byte)/(64 bits/elem)

        sx, sy, sz = self.dims[:3]

        dx = nelems / (sy * sz)
        if dx > 1:
            dx = int(dx)
            dy = sy
            dz = sz
        else:
            dx = 1
            dy = nelems / sz
            if dy > 1:
                dy = int(dy)
                dz = sz
            else:
                dy = 1
                dz = nelems

        x1, y1, z1 = self.coords
        x2, y2, z2 = (x1 + sx, y1 + sy, z1 + sz)

        for x in range(x1, x2, dx):
            for y in range(y1, y2, dy):
                for z in range(z1, z2, dz):
                    f = nib.load(self.filename)
                    chunk = Region((x, y, z),
                                   f.get_data('unchanged')[x:min(x2, x + dx), y:min(y2, y + dy), z:min(z2, z + dz)])
                    del f
                    yield chunk

    def __iter__(self):
        return self.chunks()

    def affine(self):
        f = nib.load(self.filename)
        aff = f.affine
        del f
        return aff

    def get_data(self):
        return self.niiImage.get_data()


class mghReader:
    def __init__(self, filename, x1=0, y1=0, z1=0, x2=None, y2=None, z2=None):
        self.filename = filename
        f = mghformat.MGHImage.from_filename(filename)
        self.dims = f.shape
        del f
        if x2 == None:
            x2 = self.dims[0]
        if y2 == None:
            y2 = self.dims[1]
        if z2 == None:
            z2 = self.dims[2]

        assert x1 < x2
        assert y1 < y2
        assert z1 < z2

        self.mem_usage = 0.5

        x1, y1, z1 = map(lambda a: max(a, 0), [x1, y1, z1])
        x2, y2, z2 = map(min, zip(self.dims, [x2, y2, z2]))

        self.dims = (x2 - x1, y2 - y1, z2 - z1) + self.dims[3:]
        self.coords = (x1, y1, z1)

    def chunks(self, mem_usage=None):
        if mem_usage != None:
            self.mem_usage = mem_usage

        d = 1.0
        for x in self.dims[3:]:
            d *= x
        nelems = self.mem_usage * (2 ** 17) / d  # (number of MBytes x 2**20 bytes/MB x 8 bits/byte)/(64 bits/elem)

        sx, sy, sz = self.dims[:3]

        dx = nelems / (sy * sz)
        if dx > 1:
            dx = int(dx)
            dy = sy
            dz = sz
        else:
            dx = 1
            dy = nelems / sz
            if dy > 1:
                dy = int(dy)
                dz = sz
            else:
                dy = 1
                dz = nelems

        x1, y1, z1 = self.coords
        x2, y2, z2 = (x1 + sx, y1 + sy, z1 + sz)
        for x in range(x1, x2, dx):
            for y in range(y1, y2, dy):
                for z in range(z1, z2, dz):
                    f = mghformat.MGHImage.from_filename(self.filename)
                    chunk = Region((x, y, z), f.get_data('unchanged')[x:min(x2, x + dx), y:min(y2, y + dy), z:min(z2, z + dz)])
                    del f
                    yield chunk

    def __iter__(self):
        return self.chunks()


    def affine(self):
        pass

    def get_data(self):
        return mghformat.MGHImage.from_filename(self.filename)


class SurfaceReader:
    def __init__(self, filename, x1=0, x2=None, *args, **kwargs):
        self.filename = filename
        f = read_surface(filename)

        if isinstance(f, tuple):
            self.dims = f[0].shape
        else:
            self.dims = f.shape

        print(self.dims)
        del f
        if x2 == None:
            x2 = self.dims[0]

        assert x1 < x2
        self.mem_usage = 0.5

        x1 = map(lambda a: max(a, 0), [x1])[0]
        x2 = map(min, zip(self.dims, [x2]))[0]


        self.dims = (x2 - x1,) + self.dims[1:]
        self.coords = (x1)

    def chunks(self, mem_usage=None):
        if mem_usage != None:
            self.mem_usage = mem_usage

        if self.filename.split('.')[-1] in ['inflated', 'pial', 'white']:
            raise ValueError('Chunks method not valid for [\'inflated\', \'pial\', \'white\'] file extensions.')

        d = 1.0
        for x in self.dims[1:]:
            d *= x
        nelems = self.mem_usage * (2 ** 17) / d  # (number of MBytes x 2**20 bytes/MB x 8 bits/byte)/(64 bits/elem)

        sx = self.dims[0]

        dx = nelems
        if dx > 1:
            dx = int(dx)
        else:
            dx = 1

        x1 = self.coords
        x2 = x1 + sx
        for x in range(x1, x2, dx):
            chunk = Region(x, read_surface(self.filename)[x:min(x2, x + dx)])
            yield chunk

    def __iter__(self):
        return self.chunks()

    def affine(self):
        pass

    def get_data(self):
        return read_surface(self.filename)


class NiftiWriter(niiFile):

    @staticmethod
    def open(filename):
        f = nib.load(filename)
        nw = NiftiWriter(f.get_data('unchanged'), f.affine)
        del f
        return nw

    def save(self, filename, *args, **kwargs):
        nib.save(self, filename, *args, **kwargs)

    def chunks(self, mem_usage=None, *args, **kwargs):
        try:
            return NiftiReader(self._filename, *args, **kwargs).chunks(mem_usage)
        except AttributeError:
            return ()


class SurfaceWriter(object):

    def __init__(self,data, *args, **kwargs):
        if len(data.shape) > 1:
            self.data = data
        else:
            self.data = data[:,np.newaxis]

        self.img = sitk.GetImageFromArray(self.data)

    def save(self, filepath, *args, **kwargs):

        bname = basename(filepath)
        dname = dirname(filepath)

        filename = join(dname, bname)
        sitk.WriteImage(self.img, filename)

        # io.write_morph_data(filename, self.data, *args, **kwargs)

    def chunks(self, mem_usage=None, *args, **kwargs):
        try:
            return SurfaceReader(self._filename, *args, **kwargs).chunks(mem_usage)
        except AttributeError:
            return ()


class Results(object):

    def __init__(self, loader, writer, extension):
        self.loader = loader
        self.writer = writer
        self.extension = extension



class Region:
    def __init__(self, coords, data):
        self.coords = coords
        self.data = data

    def size(self):
        return self.data.shape[:3]
