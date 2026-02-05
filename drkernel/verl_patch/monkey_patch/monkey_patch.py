import inspect
import os
from typing import List
from unittest.mock import patch

import ray
from verl.single_controller.base import Worker
from verl.single_controller.ray.base import (
    RayClassWithInitArgs,
    _bind_workers_method_to_parent,
    _unwrap_ray_remote,
)


def _get_base_class(mros: List):
    for cls in mros[0]:
        if cls.__name__ == 'Worker':
            return cls
    raise ValueError('Cannot support other base class')


def create_colocated_worker_cls_patch(class_dict: dict[str, RayClassWithInitArgs]):
    """
    This function should return a class instance that delegates the calls to every
    cls in cls_dict
    """
    cls_dict = {}
    init_args_dict = {}
    worker_cls = _get_base_class([inspect.getmro(cls.cls.__ray_actor_class__) for cls in class_dict.values()])

    assert issubclass(worker_cls, Worker), f"worker_cls {worker_cls} should be a subclass of Worker"

    for key, cls in class_dict.items():
        cls_dict[key] = cls.cls
        init_args_dict[key] = {'args': cls.args, 'kwargs': cls.kwargs}

    assert cls_dict.keys() == init_args_dict.keys()

    # TODO: create a class with customizable name
    class WorkerDict(worker_cls):

        def __init__(self):
            super().__init__()
            self.worker_dict = {}
            for key, user_defined_cls in cls_dict.items():
                user_defined_cls = _unwrap_ray_remote(user_defined_cls)
                # directly instantiate the class without remote
                # in worker class, e.g. <verl.single_controller.base.worker.Worker> when DISABLE_WORKER_INIT == 1 it will return immediately
                with patch.dict(os.environ, {'DISABLE_WORKER_INIT': '1'}):
                    self.worker_dict[key] = user_defined_cls(
                        *init_args_dict[key].get('args', ()), **init_args_dict[key].get('kwargs', {})
                    )

    # now monkey-patch the methods from inner class to WorkerDict
    for key, user_defined_cls in cls_dict.items():
        user_defined_cls = _unwrap_ray_remote(user_defined_cls)
        _bind_workers_method_to_parent(WorkerDict, key, user_defined_cls)

    remote_cls = ray.remote(WorkerDict)
    remote_cls = RayClassWithInitArgs(cls=remote_cls)
    return remote_cls


def _apply_create_colocated_worker_cls_patch():
    import verl.single_controller.ray.base

    verl.single_controller.ray.base.create_colocated_worker_cls = create_colocated_worker_cls_patch


def apply_single_controller_patches(use_new=True):
    if use_new:
        _apply_create_colocated_worker_cls_patch()
