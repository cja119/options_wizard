"""
Utils file for strategies.
"""

from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar

US_BD = CustomBusinessDay(calendar=USFederalHolidayCalendar())

