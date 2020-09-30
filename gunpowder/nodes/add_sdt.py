import logging

import numpy as np
import scipy.ndimage

from gunpowder.batch_request import BatchRequest
from gunpowder.coordinate import Coordinate
from gunpowder.ext import malis
from gunpowder.array import Array
from gunpowder.points import Points
from gunpowder.points import PointsKeys
from gunpowder.points_spec import PointsSpec
from gunpowder.roi import Roi
from .batch_filter import BatchFilter


logger = logging.getLogger(__name__)

class AddSdt(BatchFilter):
    '''Add an array with a normalized signed distance transform from the
    boundary (normalization by factor + tanh), values are computed for
    each foreground pixel.

    Args:

        labels (:class:`ArrayKey`):

            The array to read the labels from.

        threeclass (:class:`ArrayKey`):

            An array containing 3-label data, 0 for background,
            1 for inside and 2 for boundary pixel.

        sdt (:class:`ArrayKey`):

            The array to generate containing the transform.

        scale_sdt (`float`):

            A normalization factor, distance is divided by this factor.
    '''
    def __init__(
            self,
            labels,
            threeclass,
            sdt,
            scale_sdt):

        self.labels = labels
        self.threeclass = threeclass
        self.sdt = sdt
        self.scale_sdt = scale_sdt

    def setup(self):

        voxel_size = self.spec[self.threeclass].voxel_size
        spec = self.spec[self.threeclass].copy()
        spec.dtype = np.float32

        self.provides(self.sdt, spec)
        self.enable_autoskip()

    def prepare(self, request):
        deps = BatchRequest()
        deps[self.labels] = request[self.sdt].copy()
        deps[self.labels].interpolatable = False
        deps[self.labels].dtype = None
        deps[self.threeclass] = request[self.sdt].copy()
        deps[self.threeclass].interpolatable = False
        deps[self.threeclass].dtype = None
        return deps

    def process(self, batch, request):

        logger.debug("computing sdt from threeclass/labels")
        threeclass = batch.arrays[self.threeclass].data.astype(np.int32)
        labels = batch.arrays[self.labels].data

        sdt = np.zeros(threeclass.shape, dtype=np.float32)

        boundary = np.copy(threeclass)
        boundary[threeclass != 2] = 1
        boundary[threeclass == 2] = 0

        edt = scipy.ndimage.distance_transform_edt(boundary)
        sdt = np.copy(edt)
        for label in np.unique(labels):
            if label == 0:
                continue
            label_mask = labels==label
            sdt[label_mask] *= -1
        tanh =  np.tanh(1./abs(self.scale_sdt) * sdt)


        spec = batch.arrays[self.threeclass].spec.copy()
        spec.interpolatable = True
        spec.dtype = np.float32
        batch.arrays[self.sdt] = Array(tanh, spec)
