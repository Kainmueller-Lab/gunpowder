import logging

import numpy as np

from gunpowder.batch_request import BatchRequest
from gunpowder.coordinate import Coordinate
from gunpowder.ext import malis
from gunpowder.array import Array
from gunpowder.graph import Node, Graph
from gunpowder.graph_spec import GraphSpec
from gunpowder.roi import Roi
from .batch_filter import BatchFilter

logger = logging.getLogger(__name__)

class AddCPV(BatchFilter):
    '''Add an array with vectors to the instance center. Vectors are created
    for each foreground pixel.

    Args:

        points (:class:`GraphKey`):

            Graph containing the center points of all instances.
            Node ids have to match labels (use id_dim in csv points source)

        labels (:class:`ArrayKey`):

            The array to read the labels from.

        cpv (:class:`ArrayKey`):

            The array to generate containing the vectors.

        background (``int``, optional):

            Label of background pixels
    '''

    def __init__(
            self,
            points,
            labels,
            cpv,
            background=0):

        self.points = points
        self.labels = labels
        self.cpv = cpv
        self.background = background

    def setup(self):

        assert self.points in self.spec, (
            "Upstream does not provide %s needed by "
            "addCPV"%self.points)

        voxel_size = self.spec[self.labels].voxel_size

        spec = self.spec[self.labels].copy()
        pad = 30
        self.padding = Coordinate((pad,)*len(voxel_size))

        spec.dtype = np.float32

        self.provides(self.cpv, spec)
        self.enable_autoskip()

    def prepare(self, request):

        points_roi = request[self.cpv].roi.grow(
                self.padding,
                self.padding)

        logger.debug("upstream %s request: %s", self.points, points_roi)
        deps = BatchRequest()
        deps[self.points] = GraphSpec(roi=points_roi)
        deps[self.labels] = request[self.cpv].copy()
        deps[self.labels].interpolatable = False
        deps[self.labels].dtype = None
        return deps

    def process(self, batch, request):

        logger.debug("computing cpv from labels")
        arr = batch.arrays[self.labels].data.astype(np.int32)
        points = batch.points[self.points]
        if arr.shape[0] == 1:
            arr.shape = arr.shape[1:]

        cpvs = np.zeros((len(arr.shape),) + arr.shape,
                        dtype=np.float32)
        dl = batch.arrays[self.labels].spec.roi.get_offset()

        if len(arr.shape) == 2:
            for y in range(arr.shape[0]):
                for x in range(arr.shape[1]):
                    lbl = int(arr[y, x])
                    if lbl == self.background:
                        continue
                    try:
                        cntr = points.node(lbl).location
                        cpvs[0, y, x] = cntr[0] - (y + dl[0])
                        cpvs[1, y, x] = cntr[1] - (x + dl[1])
                    except KeyError:
                        logger.debug("Point %s not found in graph", lbl)
        else:
            for z in range(arr.shape[0]):
                for y in range(arr.shape[1]):
                    for x in range(arr.shape[2]):
                        lbl = int(arr[z, y, x])
                        if lbl == self.background:
                            continue
                        try:
                            cntr = points.node(lbl).location
                            cpvs[0, z, y, x] = cntr[0] - (z + dl[0])
                            cpvs[1, z, y, x] = cntr[1] - (y + dl[1])
                            cpvs[2, z, y, x] = cntr[2] - (x + dl[2])
                        except KeyError:
                            logger.debug("Point %s not found in graph", lbl)

        spec = batch.arrays[self.labels].spec.copy()
        spec.interpolatable = True
        spec.dtype = np.float32
        batch.arrays[self.cpv] = Array(cpvs, spec)
