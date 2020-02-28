"""
============================
Distribution Testing Module.
============================

This module contains tests to run to verify that your project is working
correctly.  Having a consolidated list of tests to run allows you to change
bits and pieces of your code while verifying that the changes do not break
anything.  Testing is a critical piece of any development workflow, relating
to points 2 and 3 in the Joel Spolsky's standard for good software 
development.  If you can't test your code, you can't maintain and develop it
past a certain level of complexity.  Moreover, someone else can't contribute
to your project unless you can verify that their changes are amenable.

The standard way to accomplish testing in python is via the unittest module.
To run a series of tests, at the command prompt, type
`python -m unittest -v <test_script>`.  Note that unittest is a built
in library of python.  The `-m` flag means to run the unittest module with
an argument <test_script>.  Both running a module explicitly with 
`python -m <module>` and importing it as `import <module>` from within
python execute code in the module.  However, when run as 
`python -m <module>`, the `__name__` variable is set to `'__main__'`.
This allows separate code to be executed when placed in a conditional 
block, e.g. 
```
if __name__=='__main__':
```

You will see that in the code below.  But back to the matter at hand.  
There's two approaches to testing: unit testing, which tests each 
component of your code in isolation, and integration testing, which 
tests large portions of interacting code.  The inaptly-named unittest
module handles both.

The unittest module allows a lot of flexibility in setting up and 
organizing a heirarchy of different tests.  Tests can reside in 
different modules (e.g. test1.py and test2.py), different test cases, or 
the same test case.  The example below is fairly simple; it has two test
cases: TestNeedleMethods and TestDataMethods.  These cases test every 
function of utils.py and needle.py.  

Why put do some tests go in different test cases?  Each test case can
have a setUp() and a tearDown() method.  These create resources (e.g. files)
or variables that tests in the case rely upon.  tearDown() destroys or 
releases these resources.  Within a test case, tests should be able to 
run in an arbitrary order, but can assume that setUp() has executed 
previously.  Tests that do not rely on setUp() and tearDown() should 
be in their own test case; that way, if setUp() causes an exception that
prevents that case's tests from running correctly, the test module can
still succeed at all tests in other test cases.

The crux of each test is a call to unittest.TestCase.assertX() methods.
When unittest runs, these methods allow unittest to catch and report
exceptions that occur while smoothly continuing on to the next test.

Each test case is a class that extends unittest.TestCase.  Each
test is a class method beginning with the word `test_`.  

* Python unittest module https://docs.python.org/3/library/unittest.html
* https://www.joelonsoftware.com/2000/08/09/the-joel-test-12-steps-to-better-code/


"""
# Standard library imports
import unittest
import os
# Third party imports
import pandas as pd
# Local application/library-specific imports
import seattle

class TestNeedleMethods(unittest.TestCase):
    
    def test_default_needle(self):
        attraction = seattle.needle.SeattleAttraction()
        self.assertEqual(attraction.needle, 'space needle')
    def test_default_opening(self):
        attraction = seattle.needle.SeattleAttraction()
        self.assertEqual(attraction.opening,
                         pd.to_datetime('2019 Jan 1 08:00:00').time())
    def test_default_closing(self):
        attraction = seattle.needle.SeattleAttraction()
        self.assertEqual(attraction.closing,
                         pd.to_datetime('2019 Jan 1 23:59:59').time())
    def test_close_before_opening(self):
        with self.assertRaises(seattle.utils.EndPrecedesStartError):
            seattle.needle.SeattleAttraction(opening='10:00:00', 
                                             closing='08:00:00')
    def test_early_ride(self):
        attraction = seattle.needle.SeattleAttraction()
        with self.assertRaises(seattle.needle.AfterHoursError):
            attraction.ride('06:00:00', '12:00:00')
    def test_late_ride(self):
        attraction = seattle.needle.SeattleAttraction()
        with self.assertRaises(seattle.needle.AfterHoursError):
            attraction.ride('06:00:00', '12:00:00')
    def test_backwards_ride(self):
        attraction = seattle.needle.SeattleAttraction()
        with self.assertRaises(seattle.utils.EndPrecedesStartError):
            attraction.ride('12:00:00', '11:00:00')
    def test_ok_ride(self):
        attraction = seattle.needle.SeattleAttraction('grunge')
        self.assertEqual(attraction.ride('12:00:00','13:00:00'),
                          'grunge needle')

class TestDataMethods(unittest.TestCase):
    
    def setUp(self):
        cols = "person,attraction,start,end"
        entry = "John,fire needle,2017-04-01 10:00:00, 2017-04-01 11:00:00"
#        self.data_dir = Path(__file__).parent.parent / 'data'
        data_dir = seattle.utils.data_dir
        with open(data_dir / 'bad.csv', 'w') as fh:
            fh.write(entry)
        with open(data_dir / 'test.csv', 'w') as fh:
            fh.write(cols+'\n'+entry)
            
        self.temp_df = pd.DataFrame([['Jane', 'water needle', 
                                     '2019-02-09 13:00:00', 
                                     '2019-02-09 14:00:00']],
                                    columns=['person', 'attraction', 
                                             'start', 'end'])
    def test_load(self):
        seattle.utils.load_records('test.csv')
    def test_bad_load(self):
        with self.assertRaises(AssertionError):
            seattle.utils.load_records('bad.csv')
    def test_ride(self):
        attraction = seattle.needle.SeattleAttraction()
        self.temp_df = seattle.utils.add_ride(self.temp_df, attraction, 'Jack',
                                              '2018-03-02 15:00:00',
                                              '2018-03-02 15:00:01')
    def test_save(self):
        seattle.utils.save_records(self.temp_df, 'test.csv')
    def tearDown(self):
        os.remove(seattle.utils.data_dir / 'test.csv')
        os.remove(seattle.utils.data_dir / 'bad.csv')
        pass
    
if __name__ == '__main__':
    unittest.main()