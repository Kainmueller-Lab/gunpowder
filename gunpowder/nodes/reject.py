import logging

from .batch_provider import BatchProvider
from gunpowder.profiling import Timing
from gunpowder.volume import VolumeType

logger = logging.getLogger(__name__)

class Reject(BatchProvider):

    def __init__(self, min_masked=0.5):
        self.min_masked = min_masked

    def setup(self):
        assert VolumeType.GT_MASK in self.get_spec().volumes, "Reject can only be used if GT masks are provided"
        assert len(self.get_upstream_providers()) == 1, "Reject can only be used with exactly one upstream provider."
        self.upstream_provider = self.get_upstream_providers()[0]

    def get_spec(self):
        assert len(self.get_upstream_providers()) == 1, "Reject can only be used with exactly one upstream provider."
        return self.get_upstream_providers()[0].get_spec()

    def provide(self, request):

        report_next_timeout = 10
        num_rejected = 0

        timing = Timing(self)
        timing.start()

        assert VolumeType.GT_MASK in request.volumes, "Reject can only be used if a GT mask is requested"

        have_good_batch = False
        while not have_good_batch:

            batch = self.upstream_provider.request_batch(request)
            mask_ratio = batch.volumes[VolumeType.GT_MASK].data.mean()
            have_good_batch = mask_ratio>=self.min_masked

            if not have_good_batch:

                logger.debug("reject batch with mask ratio %f at "%mask_ratio + str(batch.volumes[VolumeType.GT_MASK].roi))
                num_rejected += 1

                if timing.elapsed() > report_next_timeout:

                    logger.warning("rejected %d batches, been waiting for a good one since %ds"%(num_rejected, report_next_timeout))
                    report_next_timeout *= 2

        logger.debug("good batch with mask ratio %f found at "%mask_ratio + str(batch.volumes[VolumeType.GT_MASK].roi))

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch
