from copy import copy
import os
from enum import Enum
from typing import Any, List, Optional, TypeVar, Iterator, Iterable
import logging
import numpy as np
from robopilot.config import Config
from robopilot.parts.tub_v2 import Tub
from robopilot.utils import load_image, load_pil_image, binary_to_img, \
    img_to_arr, img_to_binary, arr_to_binary
from typing_extensions import TypedDict


logger = logging.getLogger(__name__)


class CachePolicy(Enum):
    NOCACHE = 0
    BINARY = 1
    ARRAY = 2


TubRecordDict = TypedDict(
    'TubRecordDict',
    {
        '_index': int,
        '_session_id': str,
        'cam/image_array': str,
        'user/angle': float,
        'user/throttle': float,
        'user/mode': str,
        'imu/acl_x': Optional[float],
        'imu/acl_y': Optional[float],
        'imu/acl_z': Optional[float],
        'imu/gyr_x': Optional[float],
        'imu/gyr_y': Optional[float],
        'imu/gyr_z': Optional[float],
        'behavior/one_hot_state_array': Optional[List[float]],
        'localizer/location': Optional[int]
    }
)


class TubRecord(object):
    def __init__(self, config: Config, base_path: str,
                 underlying: TubRecordDict) -> None:
        self.config = config
        self.base_path = base_path
        self.underlying = underlying
        self._cache_policy = CachePolicy[
            getattr(self.config, 'CACHE_POLICY', 'ARRAY')]
        self._cache_images = getattr(self.config, 'CACHE_IMAGES', True)
        self._image: Optional[Any] = None

    def __copy__(self):
        """ Make shallow copies of config and image and full copies of the rest.
        :return TubRecord:    TubRecord copy
        """
        tubrec = TubRecord(self.config,
                           copy(self.base_path),
                           copy(self.underlying))
        tubrec._cache_policy = copy(self._cache_policy)
        tubrec._cache_images = copy(self._cache_images)
        tubrec._image = self._image
        return tubrec

    def image(self, processor=None, as_nparray=True) -> np.ndarray:
        """
        Loads the image.

        :param processor:   Image processing like augmentations or cropping, if
                            not None. Defaults to None.
        :param as_nparray:  Whether to convert the image to a np array of uint8.
                            Defaults to True. If false, returns result of
                            Image.open()
        :return:            Image
        """
        if self._image is None:
            _image = self._extract_image(as_nparray, processor)
        else:
            _image = self._image_from_cache(as_nparray)
            if processor:
                _image = processor(_image)
        return _image

    def _image_from_cache(self, as_nparray):
        """
        Cache policy only supports numpy array format
        :return: Numpy array from cache
        """
        if not as_nparray:
            return self._image

        if self._cache_policy == CachePolicy.NOCACHE:
            raise RuntimeError("Found cached image with policy NOCACHE")
        elif self._cache_policy == CachePolicy.ARRAY:
            return self._image
        elif self._cache_policy == CachePolicy.BINARY:
            return img_to_arr(binary_to_img(self._image))
        else:
            raise RuntimeError(f"Unhandled cache policy {self._cache_policy}")

    def _load_image_and_cache(self, img_path):
        # if no caching, just load but don't cache
        if self._cache_policy == CachePolicy.NOCACHE:
            _image = load_image(img_path, cfg=self.config)
        # if caching full array, load and cache array
        elif self._cache_policy == CachePolicy.ARRAY:
            _image = load_image(img_path, cfg=self.config)
            self._image = _image
        # if caching is binary, only cache binary but return full array
        elif self._cache_policy == CachePolicy.BINARY:
            with open(img_path, 'rb') as f:
                _image = f.read()
                self._image = _image
                _image = img_to_arr(binary_to_img(_image))
        return _image

    def _load_pil_image_and_cache(self, img_path):
        _image = load_pil_image(img_path, cfg=self.config)
        if self._cache_policy != CachePolicy.NOCACHE:
            self._image = _image
        return _image

    def _cache_processed_image(self, image, as_nparray):
        if not as_nparray:
            if self._cache_policy != CachePolicy.NOCACHE:
                self._image = image
            return
        # if numpy and array caching, cache the processed image
        if self._cache_policy == CachePolicy.ARRAY:
            self._image = image
        # if numpy and binary caching, cache binary image, but return
        # numpy
        elif self._cache_policy == CachePolicy.BINARY:
            self._image = arr_to_binary(image)
        # in the case of no caching, nothing needs to be done here

    def _extract_image(self, as_nparray, processor):
        image_path = self.underlying['cam/image_array']
        full_path = os.path.join(self.base_path, 'images', image_path)
        if as_nparray:
            _image = self._load_image_and_cache(full_path)
        else:
            _image = self._load_pil_image_and_cache(full_path)
        if processor:
            # _image is now either numpy or PIL, so processing applies always
            _image = processor(_image)
            self._cache_processed_image(_image, as_nparray)
        return _image

    def __repr__(self) -> str:
        return repr(self.underlying)


class TubDataset(object):
    """
    Loads the dataset and creates a TubRecord list (or list of lists).
    """

    def __init__(self, config: Config, tub_paths: List[str],
                 seq_size: int = 0) -> None:
        self.config = config
        self.tub_paths = tub_paths
        self.tubs: List[Tub] = [Tub(tub_path, read_only=True)
                                for tub_path in self.tub_paths]
        self.records: List[TubRecord] = list()
        self.train_filter = getattr(config, 'TRAIN_FILTER', None)
        self.seq_size = seq_size

    def get_records(self):
        if not self.records:
            logger.info(f'Loading tubs from paths {self.tub_paths}')
            for tub in self.tubs:
                for underlying in tub:
                    record = TubRecord(self.config, tub.base_path, underlying)
                    if not self.train_filter or self.train_filter(record):
                        self.records.append(record)
            if self.seq_size > 0:
                seq = Collator(self.seq_size, self.records)
                self.records = list(seq)
        return self.records

    def close(self):
        for tub in self.tubs:
            tub.close()


class Collator(Iterable[List[TubRecord]]):
    """ Builds a sequence of continuous records for RNN and similar models. """
    def __init__(self, seq_length: int, records: List[TubRecord]):
        """
        :param seq_length:  length of sequence
        :param records:     input record list
        """
        self.records = records
        self.seq_length = seq_length

    @staticmethod
    def is_continuous(rec_1: TubRecord, rec_2: TubRecord) -> bool:
        """
        Checks if second record is next to first record

        :param rec_1:   first record
        :param rec_2:   second record
        :return:        if first record is followed by second record
        """
        it_is = rec_1.underlying['_index'] == rec_2.underlying['_index'] - 1 \
                and '__empty__' not in rec_1.underlying \
                and '__empty__' not in rec_2.underlying
        return it_is

    def __iter__(self) -> Iterator[List[TubRecord]]:
        """ Iterable interface. Returns a generator as Iterator. """
        it = iter(self.records)
        for this_record in it:
            seq = [this_record]
            seq_it = copy(it)
            for next_record in seq_it:
                if self.is_continuous(this_record, next_record) and \
                        len(seq) < self.seq_length:
                    seq.append(next_record)
                    this_record = next_record
                else:
                    break
            if len(seq) == self.seq_length:
                yield seq


