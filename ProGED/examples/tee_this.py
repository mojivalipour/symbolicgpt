"""Not yet finnished template for functionality like in lorenz.py file,
which uses arguments from command line for the name of log files that it
produces.
"""
import random as rand
import sys  # For cmd-line arguments.

# from IPython.utils.io import Tee  # Log results using 3th package.
from ProGED.examples.tee_so import Tee  # Log using manually copied class from a forum.

# # 0.) Log output to log_<nickname><random>.log file

def create_log(nickname="", cmd=False, with_random=True):
    """Create simple log file without starting from scratch.

    nickname -- Name your output log file.
    cmd -- Whether nickname will come from command line argument or not.
    with_random -- Add random number to filename of output log file
        to avoid overwrites or not to add.
        
    Usage:
        At the top of file just add line e.g.: 

        create_log("threshold-", with_random=False) 

        and output will be written to log file also.
    """
    log_nickname = nickname
    if cmd:
        if len(sys.argv) >= 2:
            log_nickname = sys.argv[1]
    random = str(rand.random()) if with_random else ""
    print("log id:", log_nickname + random)
    try:
        log_object = Tee("examples/log_" + log_nickname + random + "_log.txt")
    except FileNotFoundError:
        log_object = Tee("log_" + log_nickname + random + "_log.txt")

