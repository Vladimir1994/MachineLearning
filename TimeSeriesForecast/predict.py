from math import sqrt

from HoltWintersAdditive import holt_winters_additive
from HoltWintersMultiplicative import holt_winters_multiplicative


def predict(x, season_period, forecast_len):
    fa = holt_winters_additive(x, season_period, 0)
    fm = holt_winters_multiplicative(x, season_period, 0)

    rmsea = sqrt(sum([(k - n) ** 2 for k, n in zip(x, fa)]) / len(x))
    rmsem = sqrt(sum([(k - n) ** 2 for k, n in zip(x, fm)]) / len(x))

    if rmsea < rmsem:
        forecast = holt_winters_additive(x, 10, forecast_len)
        print('additive season')
    else:
        forecast = holt_winters_multiplicative(x, 10, forecast_len)
        print('multiplicative season')

    return forecast