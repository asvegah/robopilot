# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 21:27:44 2017

@author: wroscoe
"""
import os
import types
import logging

logger = logging.getLogger(__name__)


class Config:
    
    def from_pyfile(self, filename):
        d = types.ModuleType('config')
        d.__file__ = filename
        try:
            with open(filename, mode='rb') as config_file:
                exec(compile(config_file.read(), filename, 'exec'), d.__dict__)
        except IOError as e:
            e.strerror = 'Unable to load configuration file (%s)' % e.strerror
            raise
        self.from_object(d)
        return True
    
    def from_object(self, obj):
        for key in dir(obj):
            if key.isupper():
                setattr(self, key, getattr(obj, key))

    def from_dict(self, d, keys=[]):
        msg = 'Overwriting config with: '
        for k, v in d.items():
            if k.isupper():
                if k in keys or not keys:
                    setattr(self, k, v)
                    msg += f'{k}:{v}, '
        logger.info(msg)

    def __str__(self):
        result = []
        for key in dir(self):
            if key.isupper():
                result.append((key, getattr(self, key)))
        return str(result)

    def show(self):
        for attr in dir(self):
            if attr.isupper():
                print(attr, ":", getattr(self, attr))

    def to_pyfile(self, path):
        lines = []
        for attr in dir(self):
            if attr.isupper():
                v = getattr(self, attr)
                if isinstance(v, str):
                    v = f'"{v}"'
                lines.append(f'{attr} = {v}{os.linesep}')
        with open(path, 'w') as f:
            f.writelines(lines)


def load_config(config_path=None, myconfig="myconfig.py"):
    
    if config_path is None:
        main_path = os.getcwd()
        config_path = os.path.join(main_path, 'config.py')
        if not os.path.exists(config_path):
            local_config = os.path.join(os.path.curdir, 'config.py')
            if os.path.exists(local_config):
                config_path = local_config
    
    logger.info(f'loading config file: {config_path}')
    cfg = Config()
    cfg.from_pyfile(config_path)

    # look for the optional myconfig.py in the same path.
    personal_cfg_path = config_path.replace("config.py", myconfig)
    if os.path.exists(personal_cfg_path):
        logger.info(f"loading personal config over-rides from {myconfig}")
        personal_cfg = Config()
        personal_cfg.from_pyfile(personal_cfg_path)
        cfg.from_object(personal_cfg)
    else:
        logger.warning(f"personal config: file not found {personal_cfg_path}")

    return cfg
