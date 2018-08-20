from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

_KERAS_BACKEND_C8 = None
_KERAS_ENGINE_C8 = None
_KERAS_LAYERS_C8 = None
_KERAS_MODELS_C8 = None
_KERAS_UTILS_C8 = None


def set_keras_submodules(backend, engine, layers, models, utils):
    global _KERAS_BACKEND_C8
    global _KERAS_ENGINE_C8
    global _KERAS_LAYERS_C8
    global _KERAS_MODELS_C8
    global _KERAS_UTILS_C8
    _KERAS_BACKEND_C8 = backend
    _KERAS_ENGINE_C8 = engine
    _KERAS_LAYERS_C8 = layers
    _KERAS_MODELS_C8 = models
    _KERAS_UTILS_C8 = utils


submodels_def_var = {'backend' : "<module 'keras.backend' from '/usr/local/lib/python3.5/dist-packages/keras/backend/__init__.py'>",
                     'engine' : None,
                     'layers' : "<module 'keras.layers' from '/usr/local/lib/python3.5/dist-packages/keras/layers/__init__.py'>",
                     'models' : "<module 'keras.models' from '/usr/local/lib/python3.5/dist-packages/keras/models.py'>",
                     'utils' : "<module 'keras.utils' from '/usr/local/lib/python3.5/dist-packages/keras/utils/__init__.py'>"}


def get_keras_submodule(name):
    if name not in {'backend', 'engine', 'layers', 'models', 'utils'}:
        raise ImportError(
            'Can only retrieve one of "backend", '
            '"engine", "layers", "models", or "utils". '
            'Requested: %s' % name)
    """
    if _KERAS_BACKEND_C8 is None:
        raise ImportError('You need to first `import keras` '
                          'in order to use `keras_applications`. '
                          'For instance, you can do:\n\n'
                          '```\n'
                          'import keras\n'
                          'from keras_applications import vgg16\n'
                          '```\n\n'
                          'Or, preferably, this equivalent formulation:\n\n'
                          '```\n'
                          'from keras import applications\n'
                          '```\n')
    """
    return submodels_def_var[name]

