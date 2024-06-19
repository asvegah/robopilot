import math
import os
from time import time
from typing import List, Dict, Union, Tuple
import logging

from tensorflow.python.keras.models import load_model

from robopilot.config import Config
from robopilot.parts.keras import KerasPilot
from robopilot.parts.interpreter import keras_model_to_tflite, \
    saved_model_to_tensor_rt
from robopilot.pipeline.database import PilotDatabase
from robopilot.pipeline.sequence import TubRecord, TubSequence, TfmIterator
from robopilot.pipeline.types import TubDataset
from robopilot.pipeline.augmentations import ImageAugmentation
from robopilot.parts.image_transformations import ImageTransformations
from robopilot.utils import get_model_by_type, normalize_image, train_test_split
import tensorflow as tf
import numpy as np

logger = logging.getLogger(__name__)


class BatchSequence(object):
    """
    The idea is to have a shallow sequence with types that can hydrate
    themselves to np.ndarray initially and later into the types required by
    tf.data (i.e. dictionaries or np.ndarrays).
    """
    def __init__(self,
                 model: KerasPilot,
                 config: Config,
                 records: List[TubRecord],
                 is_train: bool) -> None:
        self.model = model
        self.config = config
        self.sequence = TubSequence(records)
        self.batch_size = self.config.BATCH_SIZE
        self.is_train = is_train
        self.augmentation = ImageAugmentation(config, 'AUGMENTATIONS')
        self.transformation = ImageTransformations(config, 'TRANSFORMATIONS')
        self.post_transformation = ImageTransformations(config,
                                                        'POST_TRANSFORMATIONS')
        self.pipeline = self._create_pipeline()

    def __len__(self) -> int:
        return math.ceil(len(self.pipeline) / self.batch_size)

    def image_processor(self, img_arr):
        """ Transforms the image and augments it if in training. We are not
        calling the normalisation here, because then the normalised images
        would get cached in the TubRecord, and they are 8 times larger (as
        they are 64bit floats and not uint8) """
        assert img_arr.dtype == np.uint8, \
            f"image_processor requires uint8 array but not {img_arr.dtype}"
        img_arr = self.transformation.run(img_arr)
        if self.is_train:
            img_arr = self.augmentation.run(img_arr)
        img_arr = self.post_transformation.run(img_arr)

        return img_arr

    def _create_pipeline(self) -> TfmIterator:
        """ This can be overridden if more complicated pipelines are
            required """
        # 1. Initialise TubRecord -> x, y transformations
        def get_x(record: TubRecord) -> Dict[str, Union[float, np.ndarray]]:
            """ Extracting x from record for training"""
            out_dict = self.model.x_transform(record, self.image_processor)
            # apply the normalisation here on the fly to go from uint8 -> float
            out_dict['img_in'] = normalize_image(out_dict['img_in'])
            return out_dict

        def get_y(record: TubRecord) -> Dict[str, Union[float, np.ndarray]]:
            """ Extracting y from record for training """
            y = self.model.y_transform(record)
            return y

        # 2. Build pipeline using the transformations
        pipeline = self.sequence.build_pipeline(x_transform=get_x,
                                                y_transform=get_y)
        return pipeline

    def create_tf_data(self) -> tf.data.Dataset:
        """ Assembles the tf data pipeline """
        dataset = tf.data.Dataset.from_generator(
            generator=lambda: self.pipeline,
            output_types=self.model.output_types(),
            output_shapes=self.model.output_shapes())
        return dataset.repeat().batch(self.batch_size)


def get_model_train_details(database: PilotDatabase, model: str = None) \
        -> Tuple[str, int]:
    if not model:
        model_name, model_num = database.generate_model_name()
    else:
        model_name, model_num = os.path.abspath(model), 0
    return model_name, model_num


def train(cfg: Config, tub_paths: str, model: str = None,
          model_type: str = None, transfer: str = None, comment: str = None) \
        -> tf.keras.callbacks.History:
    """
    Train the model
    """
    database = PilotDatabase(cfg)
    if model_type is None:
        model_type = cfg.DEFAULT_MODEL_TYPE
    model_path, model_num = \
        get_model_train_details(database, model)

    base_path, ext = tuple(os.path.splitext(model_path))
    kl = get_model_by_type(model_type, cfg)
    if transfer:
        kl.load(transfer)
    if cfg.PRINT_MODEL_SUMMARY:
        kl.interpreter.summary()

    tubs = tub_paths.split(',')
    all_tub_paths = [os.path.expanduser(tub) for tub in tubs]
    dataset = TubDataset(config=cfg, tub_paths=all_tub_paths,
                         seq_size=kl.seq_size())
    training_records, validation_records \
        = train_test_split(dataset.get_records(), shuffle=True,
                           test_size=(1. - cfg.TRAIN_TEST_SPLIT))
    logger.info(f'Records # Training {len(training_records)}')
    logger.info(f'Records # Validation {len(validation_records)}')
    dataset.close()

    # We need augmentation in validation when using crop / trapeze

    if 'fastai_' in model_type:
        from robopilot.parts.pytorch.torch_data \
            import TorchTubDataset, get_default_transform
        transform = get_default_transform(resize=False)
        dataset_train = TorchTubDataset(cfg, training_records, transform=transform)
        dataset_validate = TorchTubDataset(cfg, validation_records, transform=transform)
        train_size = len(training_records)
        val_size = len(validation_records)
    else:
        training_pipe = BatchSequence(kl, cfg, training_records, is_train=True)
        validation_pipe = BatchSequence(kl, cfg, validation_records, is_train=False)
        tune = tf.data.experimental.AUTOTUNE
        dataset_train = training_pipe.create_tf_data().prefetch(tune)
        dataset_validate = validation_pipe.create_tf_data().prefetch(tune)

        train_size = len(training_pipe)
        val_size = len(validation_pipe)

    assert val_size > 0, "Not enough validation data, decrease the batch " \
                         "size or add more data."
    logger.info(f'Train with image caching: '
                f'{getattr(cfg, "CACHE_IMAGES", "ARRAY")}')
    history = kl.train(model_path=model_path,
                       train_data=dataset_train,
                       train_steps=train_size,
                       batch_size=cfg.BATCH_SIZE,
                       validation_data=dataset_validate,
                       validation_steps=val_size,
                       epochs=cfg.MAX_EPOCHS,
                       verbose=cfg.VERBOSE_TRAIN,
                       min_delta=cfg.MIN_DELTA,
                       patience=cfg.EARLY_STOP_PATIENCE,
                       show_plot=cfg.SHOW_PLOT)

    # We are doing the tflite/trt conversion here on a previously saved model
    # and not on the kl.interpreter.model object directly. The reason is that
    # we want to convert the best model which is not the model in its current
    # state, but in the state it was saved the last time during training.
    if getattr(cfg, 'CREATE_TF_LITE', True):
        tf_lite_model_path = f'{base_path}.tflite'
        keras_model_to_tflite(model_path, tf_lite_model_path)

    if getattr(cfg, 'CREATE_TENSOR_RT', False):
        # convert .h5 model to .savedmodel, only if we are using h5 format
        if ext == '.h5':
            logger.info(f"Converting from .h5 to .savedmodel first")
            model_tmp = load_model(model_path, compile=False)
            # save in tensorflow savedmodel format (i.e. directory)
            model_tmp.save(f'{base_path}.savedmodel')
        # pass savedmodel to the rt converter
        saved_model_to_tensor_rt(f'{base_path}.savedmodel', f'{base_path}.trt')

    database_entry = {
        'Number': model_num,
        'Name': os.path.basename(base_path),
        'Pilot': str(kl),
        'Type': model_type,
        'Tubs': tub_paths,
        'Time': time(),
        'History': history,
        'Transfer': os.path.basename(transfer) if transfer else None,
        'Comment': comment,
        'Config': cfg.__dict__
    }
    database.add_entry(database_entry)
    database.write()

    return history
