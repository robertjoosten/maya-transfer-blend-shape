def main():
    import unittest
    import maya.standalone

    maya.standalone.initialize()

    test_runner = unittest.TextTestRunner(verbosity=3)
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover("tests", pattern="test_*.py")
    test_runner.run(test_suite)

    maya.standalone.uninitialize()


if __name__ == "__main__":
    main()
