# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
from adcp.tests import test_integration as it


class TimeSuite:
    """
    An example benchmark that times the performance of various kinds
    of iterating over dictionaries in Python.
    """

    def setup(self):
        self.d = {}
        for x in range(500):
            self.d[x] = None

    def time_standard(self):
        it.standard_sim()

    def time_keys(self):
        for key in self.d.keys():
            pass

    def time_iterkeys(self):
        for key in self.d.iterkeys():
            pass

    def time_range(self):
        d = self.d
        for key in range(500):
            x = d[key]
            if x is not None:
                print("hahahah")

    def time_xrange(self):
        d = self.d
        for key in range(500):
            x = d[key]
            if x is not None:
                print("hahahah")


class MemSuite:
    def mem_list(self):
        return [0] * 256
