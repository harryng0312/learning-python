# This is a sample Python script.
import sys
import calendar
import time
from datetime import datetime

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def test_iterator():
    ls = [1, 2, 3, 4]
    it = iter(ls)  # this builds an iterator object
    print(next(it))  # prints next available element in iterator Iterator object can be traversed using regular for
    # statement !usr/bin/python3
    # for x in it:
    #     print(x, end=" ")

    while True:
        try:
            print(next(it))
        except StopIteration:
            sys.exit()  # you have to import sys module for this


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    cal = calendar.month(2016, 2)
    dateTimeObj = datetime.now()
    print("Here is time: ", time.ctime(100_000))
    print("Here is timezone: ", time.timezone)
    print("Here is the calendar:")
    print(cal)
    print(f"time: " + dateTimeObj.strftime("%d-%b-%Y (%H:%M:%S.%f)"))
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    # test_iterator()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
