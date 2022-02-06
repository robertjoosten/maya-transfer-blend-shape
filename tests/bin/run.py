def main():
    import sys
    import unittest
    import maya.standalone

    maya.standalone.initialize()

    test_runner = unittest.TextTestRunner(verbosity=2)
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover("tests", pattern="test_*.py")
    result = test_runner.run(test_suite)

    maya.standalone.uninitialize()
    sys.exit(not result.wasSuccessful())


if __name__ == "__main__":
    main()
