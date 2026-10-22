import sys
import unittest
sys.argv = ['unittest']
try:
    from tests.decoupled_heloco.test_transport import TransportTests
    suite = unittest.TestLoader().loadTestsFromTestCase(TransportTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        sys.exit(1)
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
