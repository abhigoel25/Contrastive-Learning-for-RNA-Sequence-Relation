"""Utilities for dealing with collection objects (lists, dicts) and configs.
    Copied from "https://github.com/kuleshov-group/caduceus"

"""

import functools
from typing import Sequence, Mapping, Callable
import logging
import warnings
import hydra
import json
from omegaconf import ListConfig, DictConfig,OmegaConf
import rich.syntax
import rich.tree
from lightning.pytorch.utilities import rank_zero_only


# TODO this is usually used in a pattern where it's turned into a list, so can just do that here
def is_list(x):
    return isinstance(x, Sequence) and not isinstance(x, str)

# === Automatic path resolver for 'Contrastive_Learning' ===
# from pathlib import Path

# def find_contrastive_root(start_path: Path = Path(__file__)) -> Path:
#     for parent in start_path.resolve().parents:
#         if parent.name == "Contrastive_Learning":
#             return parent
#     raise RuntimeError("Could not find 'Contrastive_Learning' directory.")

# # Register resolver ONCE
# if not OmegaConf.has_resolver("contrastive_root"):
#     CONTRASTIVE_ROOT = find_contrastive_root()
#     OmegaConf.register_new_resolver("contrastive_root", lambda: str(CONTRASTIVE_ROOT))


def is_dict(x):
    return isinstance(x, Mapping)


def to_dict(x, recursive=True):
    """Convert Sequence or Mapping object to dict

    lists get converted to {0: x[0], 1: x[1], ...}
    """
    if is_list(x):
        x = {i: v for i, v in enumerate(x)}
    if is_dict(x):
        if recursive:
            return {k: to_dict(v, recursive=recursive) for k, v in x.items()}
        else:
            return dict(x)
    else:
        return x


def to_list(x, recursive=False):
    """Convert an object to list.

    If Sequence (e.g. list, tuple, Listconfig): just return it

    Special case: If non-recursive and not a list, wrap in list
    """
    if is_list(x):
        if recursive:
            return [to_list(_x) for _x in x]
        else:
            return list(x)
    else:
        if recursive:
            return x
        else:
            return [x]


def extract_attrs_from_obj(obj, *attrs):
    if obj is None:
        assert len(attrs) == 0
        return []
    return [getattr(obj, attr, None) for attr in attrs]


def auto_assign_attrs(cls, **kwargs):
    for k, v in kwargs.items():
        setattr(cls, k, v)
        
        
def instantiate(registry, config, *args, partial=False, wrap=None, **kwargs):
    """
    registry: Dictionary mapping names to functions or target paths (e.g. {'model': 'models.SequenceModel'})
    config: Dictionary with a '_name_' key indicating which element of the registry to grab, and kwargs to be passed into the target constructor
    wrap: wrap the target class (e.g. ema optimizer or tasks.wrap)
    *args, **kwargs: additional arguments to override the config to pass into the target constructor
    """

    # Case 1: no config
    if config is None:
        return None
    # Case 2a: string means _name_ was overloaded
    if isinstance(config, str):
        _name_ = None
        _target_ = registry[config]
        config = {}
    # Case 2b: grab the desired callable from name
    else:
        _name_ = config.pop("_name_")
        _target_ = registry[_name_]

    # Retrieve the right constructor automatically based on type
    if isinstance(_target_, str):
        fn = hydra.utils.get_method(path=_target_)
    elif isinstance(_target_, Callable):
        fn = _target_
    else:
        raise NotImplementedError("instantiate target must be string or callable")

    # Instantiate object
    if wrap is not None:
        fn = wrap(fn)
    obj = functools.partial(fn, *args, **config, **kwargs)

    # Restore _name_
    if _name_ is not None:
        config["_name_"] = _name_

    if partial:
        return obj
    else:
        return obj()


def get_class(registry, _name_):
    return hydra.utils.get_class(path=registry[_name_])


def omegaconf_filter_keys(d, fn=None):
    """Only keep keys where fn(key) is True. Support nested DictConfig.
    # TODO can make this inplace?
    """
    if fn is None:
        fn = lambda _: True
    if is_list(d):
        return ListConfig([omegaconf_filter_keys(v, fn) for v in d])
    elif is_dict(d):
        return DictConfig(
            {k: omegaconf_filter_keys(v, fn) for k, v in d.items() if fn(k)}
        )
    else:
        return d

def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger.x"""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger

def process_config(config: DictConfig) -> DictConfig:  # TODO because of filter_keys, this is no longer in place
    """A couple of optional utilities, controlled by main config file:
    - disabling warnings
    - easier access to debug mode
    - forcing debug friendly configuration
    Modifies DictConfig in place.
    Args:
        config (DictConfig): Configuration composed by Hydra.
    """
    log = get_logger()

    # Filter out keys that were used just for interpolation
    config = omegaconf_filter_keys(config, lambda k: not k.startswith('__'))

    # enable adding new keys to config
    OmegaConf.set_struct(config, False)

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    if config.get("debug"):
        log.info("Running in debug mode! <config.debug=True>")
        config.trainer.fast_dev_run = True

        # force debugger friendly configuration
        log.info("Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>")
        # Debuggers don't like GPUs or multiprocessing
        if config.trainer.get("gpus"):
            config.trainer.gpus = 0
        if config.loader.get("pin_memory"):
            config.loader.pin_memory = False
        if config.loader.get("num_workers"):
            config.loader.num_workers = 0

    # disable adding new keys to config
    # OmegaConf.set_struct(config, True) # [21-09-17 AG] I need this for .pop(_name_) pattern among other things

    return config,log


@rank_zero_only
def print_config(
        config: DictConfig,
        resolve: bool = True,
        save_cfg=True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.
    Args:
        config (DictConfig): Configuration composed by Hydra.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
        save_cfg (bool, optional): Whether to save the config to a file.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    fields = config.keys()
    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    if save_cfg:
        with open("config_tree.txt", "w") as fp:
            rich.print(tree, file=fp)
        with open("model_config.json", "w") as fp:  # Save config / model config for use in fine-tuning or testing
            model_config = {
                k: v
                for k, v in OmegaConf.to_container(config.embedder, resolve=True).items()
                if not k.startswith("_") or k == "config_path"
            }
            json.dump(model_config, fp, indent=4)
        with open("config.json", "w") as fp:
            json.dump(OmegaConf.to_container(config, resolve=True), fp, indent=4)