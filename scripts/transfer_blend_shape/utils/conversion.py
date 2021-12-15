def as_chunks(l, num):
    """
    :param list l:
    :param int num: Size of split
    :return: Split list
    :rtype: list
    """
    chunks = []
    for i in range(0, len(l), num):
        chunks.append(l[i:i + num])
    return chunks
