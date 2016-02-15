from math import sqrt

from HoltWintersAdditive import HoltWintersAdditive
from HoltWintersMultiplicative import HoltWintersMultiplicative


def predict(x, seasonPeriod, forecastLen):
    fa = HoltWintersAdditive(x, seasonPeriod, 0)
    fm = HoltWintersMultiplicative(x, seasonPeriod, 0)

    rmsea = sqrt(sum([(k - n) ** 2 for k, n in zip(x, fa)]) / len(x))
    rmsem = sqrt(sum([(k - n) ** 2 for k, n in zip(x, fm)]) / len(x))

    if rmsea < rmsem:
        forecast = HoltWintersAdditive(x, 10, forecastLen)
        print('additive season')
    else:
        forecast = HoltWintersMultiplicative(x, 10, forecastLen)
        print('multiplicative season')

    return forecast