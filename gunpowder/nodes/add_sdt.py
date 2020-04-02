import logging
import numpy as np
import scipy.ndimage

from .batch_filter import BatchFilter
from gunpowder.coordinate import Coordinate
from gunpowder.array import Array
from gunpowder.points import Points
from gunpowder.points import PointsKeys
from gunpowder.points_spec import PointsSpec
from gunpowder.roi import Roi


logger = logging.getLogger(__name__)

class AddSdt(BatchFilter):
    def __init__(
            self,
            labels,
            threeclass,
            sdt,
            scale_sdt):

        self.threeclass = threeclass
        self.labels = labels
        self.sdt = sdt
        self.scale_sdt = scale_sdt

    def setup(self):

        voxel_size = self.spec[self.threeclass].voxel_size
        spec = self.spec[self.threeclass].copy()
        spec.dtype = np.float32

        self.provides(self.sdt, spec)
        self.enable_autoskip()

    # def prepare(self, request):
    #     pass

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
        # print(spec)
        spec.interpolatable = True
        spec.dtype = np.float32
        # print(spec)
        batch.arrays[self.sdt] = Array(tanh, spec)
