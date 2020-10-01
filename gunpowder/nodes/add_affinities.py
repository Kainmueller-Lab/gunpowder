import logging
import numpy as np

from .batch_filter import BatchFilter
from gunpowder.array import Array
from gunpowder.batch_request import BatchRequest
from gunpowder.coordinate import Coordinate
from gunpowder.ext import malis

logger = logging.getLogger(__name__)


def seg_to_affgraph_2d(seg, nhood):
    # constructs a 2d affinity graph from a segmentation
    # assume affinity graph is represented as:
    # shape = (e, y, x)
    # nhood.shape = (edges, 2)
    shape = seg.shape
    nEdge = nhood.shape[0]

    aff = np.zeros((nEdge,)+shape,dtype=np.int32)

    for e in range(nEdge):
        # first == second pixel?
        tt1 = seg[
            max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]),
            max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1])]
        tt2 = seg[
            max(0,nhood[e,0]):min(shape[0],shape[0]+nhood[e,0]),
            max(0,nhood[e,1]):min(shape[1],shape[1]+nhood[e,1])]
        t1 = tt1 == tt2
        # first pixel fg?
        t2 = seg[
            max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]),
            max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1])] > 0
        # second pixel fg?
        t3 = seg[
            max(0,nhood[e,0]):min(shape[0],shape[0]+nhood[e,0]),
            max(0,nhood[e,1]):min(shape[1],shape[1]+nhood[e,1])] > 0
        aff[e,
            max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]),
            max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1])] = t1 * t2 * t3

    return aff

def seg_to_affgraph_2d_multi(seg, nhood):
    # constructs a 2d affinity graph from a segmentation
    # there can be multiple labels per pixel
    # assume affinity graph is represented as:
    # shape = (e, y, x)
    # nhood.shape = (edges, 2)
    shape = seg.shape[1:]
    nEdge = nhood.shape[0]

    aff = np.zeros((nEdge,)+shape,dtype=np.int32)

    for e in range(nEdge):
        # first == second pixel?
        tt1 = seg[
            :,
            max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]),
            max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1])]
        tt2 = seg[
            :,
            max(0,nhood[e,0]):min(shape[0],shape[0]+nhood[e,0]),
            max(0,nhood[e,1]):min(shape[1],shape[1]+nhood[e,1])]
        t1 = tt1 == tt2
        t1[tt1==0] = False
        t1 = np.any(t1, axis=0, keepdims=True)

        # first pixel fg?
        t2 = np.any(seg[
            :,
            max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]),
            max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1])],
                    axis=0)

        # second pixel fg?
        t3 = np.any(seg[
            :,
            max(0,nhood[e,0]):min(shape[0],shape[0]+nhood[e,0]),
            max(0,nhood[e,1]):min(shape[1],shape[1]+nhood[e,1])],
                    axis=0)

        aff[e,
            max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
            max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1])] = t1 * t2 * t3

    return aff

def seg_to_affgraph_2d_cs(seg, nhood, cs_id=None):
    # constructs a 2d affinity graph from a segmentation for cityscapes
    # assume affinity graph is represented as:
    # shape = (e, y, x)
    # nhood.shape = (edges, 2)
    shape = seg.shape
    nEdge = nhood.shape[0]

    aff = np.zeros((nEdge,)+shape,dtype=np.int32)

    for e in range(nEdge):
        # first == second pixel?
        tt1 = seg[
            max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]),
            max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1])]
        tt2 = seg[
            max(0,nhood[e,0]):min(shape[0],shape[0]+nhood[e,0]),
            max(0,nhood[e,1]):min(shape[1],shape[1]+nhood[e,1])]
        t1 = tt1 == tt2

        if cs_id is not None:
            # first pixel fg?
            t2 = np.logical_and(
                seg[max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]),
                    max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1])] >= cs_id,
                seg[max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]),
                    max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1])] < cs_id + 1000)
            # second pixel fg?
            t3 = np.logical_and(
                seg[max(0,nhood[e,0]):min(shape[0],shape[0]+nhood[e,0]),
                    max(0,nhood[e,1]):min(shape[1],shape[1]+nhood[e,1])] >= cs_id,
                seg[max(0,nhood[e,0]):min(shape[0],shape[0]+nhood[e,0]),
                    max(0,nhood[e,1]):min(shape[1],shape[1]+nhood[e,1])] < cs_id + 1000)
        else:
            # first pixel fg?
            t2 = seg[max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]),
                     max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1])] > 255
            # second pixel fg?
            t3 = seg[max(0,nhood[e,0]):min(shape[0],shape[0]+nhood[e,0]),
                     max(0,nhood[e,1]):min(shape[1],shape[1]+nhood[e,1])] > 255
        aff[e,
            max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]),
            max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1])] = t1 * t2 * t3

    return aff


class AddAffinities(BatchFilter):
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
            affinity_neighborhood,
            labels,
            affinities,
            labels_mask=None,
            unlabelled=None,
            affinities_mask=None,
            multiple_labels=False,
            cityscape=False,
            cs_id=None,
            dtype=np.uint8):

        self.affinity_neighborhood = np.array(affinity_neighborhood)
        self.labels = labels
        self.unlabelled = unlabelled
        self.labels_mask = labels_mask
        self.affinities = affinities
        self.affinities_mask = affinities_mask
        self.multiple_labels = multiple_labels
        self.cityscape = cityscape
        self.cs_id = cs_id
        self.dtype = dtype

    def setup(self):

        assert self.labels in self.spec, (
            "Upstream does not provide %s needed by "
            "AddAffinities"%self.labels)

        voxel_size = self.spec[self.labels].voxel_size

        dims = self.affinity_neighborhood.shape[1]
        self.padding_neg = Coordinate(
                min([0] + [a[d] for a in self.affinity_neighborhood])
                for d in range(dims)
        )*voxel_size

        self.padding_pos = Coordinate(
                max([0] + [a[d] for a in self.affinity_neighborhood])
                for d in range(dims)
        )*voxel_size

        logger.debug("padding neg: " + str(self.padding_neg))
        logger.debug("padding pos: " + str(self.padding_pos))

        spec = self.spec[self.labels].copy()
        if spec.roi is not None:
            spec.roi = spec.roi.grow(self.padding_neg, -self.padding_pos)
        spec.dtype = self.dtype

        self.provides(self.affinities, spec)
        if self.affinities_mask:
            self.provides(self.affinities_mask, spec)
        self.enable_autoskip()

    def prepare(self, request):

        if self.labels_mask:
            assert (
                request[self.labels].roi ==
                request[self.labels_mask].roi),(
                "requested GT label roi %s and GT label mask roi %s are not "
                "the same."%(
                    request[self.labels].roi,
                    request[self.labels_mask].roi))

        if self.unlabelled:
            assert (
                request[self.labels].roi ==
                request[self.unlabelled].roi),(
                "requested GT label roi %s and GT unlabelled mask roi %s are not "
                "the same."%(
                    request[self.labels].roi,
                    request[self.unlabelled].roi))

        deps = BatchRequest()

        # grow labels ROI to accomodate padding
        labels_roi = request[self.affinities].roi.grow(
            -self.padding_neg,
            self.padding_pos)
        deps[self.labels] = request[self.affinities].copy()
        deps[self.labels].dtype = None
        deps[self.labels].roi = labels_roi

        if self.labels_mask:
            deps[self.labels_mask] = deps[self.labels].copy()
        if self.unlabelled:
            deps[self.unlabelled] = deps[self.labels].copy()

        return deps

    def process(self, batch, request):

        affinities_roi = request[self.affinities].roi

        logger.debug("computing ground-truth affinities from labels")

        # remove superfluous channel dim if it exists
        arr = batch.arrays[self.labels].data.astype(np.int32)
        if arr.shape[0] == 1:
            arr.shape = arr.shape[1:]

        if self.cityscape:
            affinities = seg_to_affgraph_2d_cs(
                arr,
                self.affinity_neighborhood,
                self.cs_id)
        else:
            if self.multiple_labels and len(arr.shape) == 3:
                seg_to_affgraph_fun = seg_to_affgraph_2d_multi
            elif len(arr.shape) == 2:
                seg_to_affgraph_fun = seg_to_affgraph_2d
            elif self.multiple_labels and len(arr.shape) == 4:
                raise NotImplementedError
            else:
                seg_to_affgraph_fun = malis.seg_to_affgraph

            affinities = seg_to_affgraph_fun(
                arr,
                self.affinity_neighborhood)

        affinities = affinities.astype(self.dtype)

        # crop affinities to requested ROI
        offset = affinities_roi.get_offset()
        shift = -offset - self.padding_neg
        crop_roi = affinities_roi.shift(shift)
        crop_roi /= self.spec[self.labels].voxel_size
        crop = crop_roi.get_bounding_box()

        logger.debug("cropping with " + str(crop))
        affinities = affinities[(slice(None),)+crop]

        spec = self.spec[self.affinities].copy()
        spec.roi = affinities_roi
        batch.arrays[self.affinities] = Array(affinities, spec)

        if self.affinities_mask and self.affinities_mask in request:

            if self.labels_mask:

                logger.debug("computing ground-truth affinities mask from "
                             "labels mask")
                affinities_mask = malis.seg_to_affgraph(
                    batch.arrays[self.labels_mask].data.astype(np.int32),
                    self.affinity_neighborhood)
                affinities_mask = affinities_mask[(slice(None),)+crop]

            else:

                affinities_mask = np.ones_like(affinities)

            if self.unlabelled:

                # 1 for all affinities between unlabelled voxels
                unlabelled = (1 - batch.arrays[self.unlabelled].data)
                unlabelled_mask = malis.seg_to_affgraph(
                    unlabelled.astype(np.int32),
                    self.affinity_neighborhood)
                unlabelled_mask = unlabelled_mask[(slice(None),)+crop]

                # 0 for all affinities between unlabelled voxels
                unlabelled_mask = (1 - unlabelled_mask)

                # combine with mask
                affinities_mask = affinities_mask*unlabelled_mask

            affinities_mask = affinities_mask.astype(np.float32)
            batch.arrays[self.affinities_mask] = Array(affinities_mask, spec)

        else:

            if self.labels_mask is not None:
                logger.warning("GT labels does have a mask, but affinities "
                               "mask is not requested.")

        # crop labels to original label ROI
        if self.labels in request:
            roi = request[self.labels].roi
            batch.arrays[self.labels] = batch.arrays[self.labels].crop(roi)

        # same for label mask
        if self.labels_mask and self.labels_mask in request:
            roi = request[self.labels_mask].roi
            batch.arrays[self.labels_mask] = \
                batch.arrays[self.labels_mask].crop(roi)

        # and unlabelled mask
        if self.unlabelled and self.unlabelled in request:
            roi = request[self.unlabelled].roi
            batch.arrays[self.unlabelled] = \
                batch.arrays[self.unlabelled].crop(roi)

        batch.affinity_neighborhood = self.affinity_neighborhood
