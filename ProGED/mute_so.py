"""This file redirects standard output stdout, also on the file 
descriptors level (hardcoded-like output).

It is usefull and used to mute, i.e. suppress output from LSODA 
solver from ODEPACK written in FORTRAN, which often writes warnning
messages in such a hard way. In that case output equals os.devnull i.e.
it goes to nowhere.
Source code is copied from:
https://stackoverflow.com/questions/4675728/redirect-stdout-to-a-file-in-python/22434262#22434262
"""
import os
import sys
from contextlib import contextmanager

def fileno(file_or_fd):
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd

@contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    # print(stdout, type(stdout))
    if stdout is None:
       stdout = sys.stdout
    # print("still ok")
    # print(stdout, type(stdout))
    # print(sys.stdout, type(sys.stdout))

    stdout_fd = fileno(stdout)
    # print("after fileno(stdout)")
    # copy stdout_fd before it is overwritten
    #NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied: 
        
        # print("inside with")
        stdout.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            # print("excepted ValueError")
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout # allow code to be run with the redirected stdout
            # print("yield tried")
        finally:
            # print("finnaly")
            # restore stdout to its previous value
            #NOTE: dup2 makes stdout_fd inheritable unconditionally
            stdout.flush()
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied
            # print("finnaly end")

# The same example works now if stdout_redirected() is used instead of redirect_stdout():

# import os
# import sys

# stdout_fd = sys.stdout.fileno()
# with open('output.txt', 'w') as f, stdout_redirected(f):
#     print('redirected to a file')
#     os.write(stdout_fd, b'it is redirected now\n')
#     os.system('echo this is also redirected')
# print('this is goes back to stdout')
