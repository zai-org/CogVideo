import logging
import random

from torch.utils.data import Dataset, Sampler


logger = logging.getLogger(__name__)


class BucketSampler(Sampler):
    r"""
    PyTorch Sampler that groups 3D data by height, width and frames.

    Args:
        data_source (`VideoDataset`):
            A PyTorch dataset object that is an instance of `VideoDataset`.
        batch_size (`int`, defaults to `8`):
            The batch size to use for training.
        shuffle (`bool`, defaults to `True`):
            Whether or not to shuffle the dataset order and each batch before
            dispatching to the dataloader.
        drop_last (`bool`, defaults to `False`):
            Whether or not to drop incomplete buckets of data after completely iterating over all data
            in the dataset. If set to True, only batches that have `batch_size` number of entries will
            be yielded. If set to False, it is guaranteed that all data in the dataset will be processed
            and batches that do not have `batch_size` number of entries will also be yielded.
    """

    def __init__(
        self,
        data_source: Dataset,
        batch_size: int = 8,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.buckets = {resolution: [] for resolution in data_source.video_resolution_buckets}

        self._raised_warning_for_drop_last = False

    def __len__(self):
        if self.drop_last and not self._raised_warning_for_drop_last:
            self._raised_warning_for_drop_last = True
            logger.warning(
                "Calculating the length for bucket sampler is not possible when `drop_last` is set to True. This may cause problems when setting the number of epochs used for training."
            )
        return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        # Buckets hold samples between yields.  They must be recreated for
        # every epoch: with ``drop_last=True`` an incomplete bucket is
        # intentionally discarded and must not be mixed with samples from the
        # next call to ``__iter__``.
        self.buckets = {resolution: [] for resolution in self.data_source.video_resolution_buckets}

        # Shuffling individual buckets is not sufficient because the dataset
        # is otherwise read in index order.  A global permutation changes the
        # order in which buckets are filled while retaining homogeneous
        # batches, which gives each epoch a genuinely different sample order.
        indices = list(range(len(self.data_source)))
        if self.shuffle:
            random.shuffle(indices)

        for index in indices:
            data = self.data_source[index]
            video_metadata = data["video_metadata"]
            f, h, w = (
                video_metadata["num_frames"],
                video_metadata["height"],
                video_metadata["width"],
            )

            self.buckets[(f, h, w)].append(data)
            if len(self.buckets[(f, h, w)]) == self.batch_size:
                if self.shuffle:
                    random.shuffle(self.buckets[(f, h, w)])
                yield self.buckets[(f, h, w)]
                del self.buckets[(f, h, w)]
                self.buckets[(f, h, w)] = []

        if self.drop_last:
            return

        for fhw, bucket in list(self.buckets.items()):
            if len(bucket) == 0:
                continue
            if self.shuffle:
                random.shuffle(bucket)
                yield bucket
                del self.buckets[fhw]
                self.buckets[fhw] = []
