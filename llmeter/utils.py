# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


class DeferredError:
    """Stores an exception and raises it at a later time if this object is accessed in any way.

    Useful to allow soft-dependencies on imports, so that the ImportError can be raised again
    later if code actually relies on the missing library.
    Lifted from https://github.com/aws/sagemaker-python-sdk/blob/e626647692d136155d72cf5b100043af41ab6c43/src/sagemaker/utils.py#L788

    Example::

        try:
            import obscurelib
        except ImportError as e:
            logger.warning("Failed to import obscurelib. Obscure features will not work.")
            obscurelib = DeferredError(e)
    """

    def __init__(self, exception):
        self.exc = exception

    def __getattr__(self, name):
        """Called by Python interpreter before using any method or property on the object.

        So this will short-circuit essentially any access to this object.

        Args:
            name:
        """
        raise self.exc
