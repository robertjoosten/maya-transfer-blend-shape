def get_name(node_name):
    """
    Omit any parenting information from the provided node name.

    :param str node_name:
    :return: Name
    :rtype: str
    """
    return node_name.rsplit("|", 1)[-1]


def get_leaf_name(node_name):
    """
    Omit the parenting and namespace information from a provided node name.

    :param str node_name:
    :return: Leaf name
    :rtype: str
    """
    return get_name(node_name).rsplit(":", 1)[-1]