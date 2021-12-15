from maya import cmds


def main():
    from transfer_blend_shape import install
    install.execute()


if not cmds.about(batch=True):
    cmds.evalDeferred(main)
