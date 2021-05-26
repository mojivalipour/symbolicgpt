"""Solution class for copying output to log file. Copied from:
https://stackoverflow.com/questions/616645/how-to-duplicate-sys-stdout-to-a-log-file/616686#616686
"""

import sys

class Tee(object):
    def __init__(self, name, mode="w"):
        self.file = open(name, mode, encoding='utf-8')
        self.stdout = sys.stdout
        sys.stdout = self
    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()
        # print("Destructor __del__ was called inside SO's Tee.")
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    def flush(self):
        self.file.flush()
        self.stdout.flush()

class TeeFileOnly(object):
    def __init__(self, name, mode="w"):
        self.file = open(name, mode, encoding='utf-8')
        self.stdout = sys.stdout
        sys.stdout = self
    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()
        # print("Destructor __del__ was called inside SO's Tee.")
    def write(self, data):
        self.file.write(data)
    def flush(self):
        self.file.flush()

# For muting also file descriptors, i.e. lower level writes, see mute_so.py.
# Here is only naive solution:
class Mute(object):
    def __init__(self):
        self.stdout = sys.stdout
        sys.stdout = self
    def __del__(self):
        sys.stdout = self.stdout
    def write(self, data):
        pass
    def flush(self):
        pass
        
