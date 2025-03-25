import time

def timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))

def timeprint(logstr):
    print(f"{timestamp()} {logstr}")