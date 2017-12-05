#! /usr/bin/env python3
# 05 11 20:09
from pytocl.main import main
from my_driver import MyDriver

if __name__ == '__main__':
    main(MyDriver(logdata=False))
