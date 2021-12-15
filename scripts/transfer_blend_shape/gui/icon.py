import os
import logging

from transfer_blend_shape.utils import decorator


log = logging.getLogger(__name__)


@decorator.memoize
def get_icon_file_path(file_name):
    """
    :return: Icon file path
    :rtype: str
    """
    icon_directories = os.environ.get("XBMLANGPATH")
    icon_directories = icon_directories.split(os.pathsep) if icon_directories else []

    for directory in icon_directories:
        file_path = os.path.join(directory, file_name)
        if os.path.exists(file_path):
            return file_path

    log.debug("File '{}' not found in {}.".format(file_name, icon_directories))
    return ":/{}".format(file_name)
