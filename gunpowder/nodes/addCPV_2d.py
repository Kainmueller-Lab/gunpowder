import logging
import numpy as np

from .batch_filter import BatchFilter
from gunpowder.coordinate import Coordinate
from gunpowder.ext import malis
from gunpowder.array import Array
from gunpowder.points import Points
from gunpowder.points import PointsKeys
from gunpowder.points_spec import PointsSpec
from gunpowder.roi import Roi


logger = logging.getLogger(__name__)

class AddCPV_2d(BatchFilter):
    '''Add an array with affinities for a given label array and neighborhood to 
    the batch. Affinity values are created one for each voxel and entry in the 
    neighborhood list, i.e., for each voxel and each neighbor of this voxel. 
    Values are 1 iff both labels (of the voxel and the neighbor) are equal and 
    non-zero.

    Args:

        affinity_neighborhood (``list`` of array-like):

            List of offsets for the affinities to consider for each voxel.

        labels (:class:`ArrayKey`):

            The array to read the labels from.

        affinities (:class:`ArrayKey`):

            The array to generate containing the affinities.

        labels_mask (:class:`ArrayKey`, optional):

            The array to use as a mask for ``labels``. Affinities connecting at
            least one masked out label will be masked out in
            ``affinities_mask``. If not given, ``affinities_mask`` will contain
            ones everywhere (if requested).

        unlabelled (:class:`ArrayKey`, optional):

            A binary array to indicate unlabelled areas with 0. Affinities from
            labelled to unlabelled voxels are set to 0, affinities between
            unlabelled voxels are masked out (they will not be used for
            training).

        affinities_mask (:class:`ArrayKey`, optional):

            The array to generate containing the affinitiy mask, as derived
            from parameter ``labels_mask``.
    '''

    def __init__(
            self,
            points,
            labels,
            cpv):

        self.points = points
        self.cpv = cpv
        self.labels = labels

    def setup(self):

        assert self.points in self.spec, (
            "Upstream does not provide %s needed by "
            "addCPV"%self.points)

        voxel_size = self.spec[self.labels].voxel_size

        spec = self.spec[self.labels].copy()
        pad = 30
        self.padding = Coordinate((pad,pad))

        # spec.roi = spec.roi.grow(self.padding, -self.padding)
        spec.dtype = np.float32

        self.provides(self.cpv, spec)
        self.enable_autoskip()

    def prepare(self, request):

        # if self.points not in request:
            # request[self.points] = request[self.labels].copy()
        points_roi = request[self.labels].roi.grow(
                self.padding,
                self.padding)
        request[self.points] = PointsSpec(roi=points_roi)

        # labels_roi = request[self.labels].roi
        # context_roi = request[self.affinities].roi.grow(
        #     -self.padding,
        #     self.padding)

        # # grow labels ROI to accomodate padding
        # labels_roi = labels_roi.union(context_roi)
        # request[self.labels].roi = labels_roi

        logger.debug("upstream %s request: "%self.points + str(points_roi))

    def process(self, batch, request):

        logger.debug("computing cpv from labels")
        arr = batch.arrays[self.labels].data.astype(np.int32)
        points = batch.points[self.points].data
        if arr.shape[0] == 1:
            arr.shape = arr.shape[1:]

        cpvs = np.zeros((2, arr.shape[0], arr.shape[1]),
                        dtype=np.float32)
        dly = batch.arrays[self.labels].spec.roi.get_offset()[0]
        dlx = batch.arrays[self.labels].spec.roi.get_offset()[1]

        for y in range(arr.shape[0]):
            for x in range(arr.shape[1]):
                lbl = int(arr[y,x])
                if lbl in points:
                    cntr = points[lbl].location
                    cpvs[0,y,x] = (cntr[0])-(y+dly)
                    cpvs[1,y,x] = (cntr[1])-(x+dlx)


        spec = batch.arrays[self.labels].spec.copy()
        # print(spec)
        spec.interpolatable = True
        spec.dtype = np.float32
        # print(spec)
        batch.arrays[self.cpv] = Array(cpvs, spec)
