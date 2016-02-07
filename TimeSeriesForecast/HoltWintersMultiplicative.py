from scipy.optimize import minimize
from numpy import array
from rmseHoltWintersMultiplicative import rmseHoltWintersMultiplicative


def HoltWintersMultiplicative(x, seasonPeriod, forecastLen):
    Y = x[:]
    initValues = array([0.5, 0.5, 0.5])
    boundaries = [(0, 1), (0, 1), (0, 1)]

    parameters = minimize(rmseHoltWintersMultiplicative, x0=initValues,
                          args=(Y, seasonPeriod), bounds=boundaries)
    alpha, beta, gamma = parameters.x

    level = [sum(Y[0:seasonPeriod]) / float(seasonPeriod)]
    trend = [(sum(Y[seasonPeriod:2 * seasonPeriod]) - sum(Y[0:seasonPeriod]))
             / seasonPeriod ** 2]
    season = [Y[i] / (level[0] + trend[0]) for i in range(seasonPeriod)]
    forecast = [(level[0] + trend[0]) * season[0]]

    for i in range(len(Y) + forecastLen):
        if i == len(Y):
            Y.append((level[-1] + trend[-1]) * season[-seasonPeriod])

        if(season[i] < 10 ** 8):
            level.append(alpha * (Y[i]) + (1 - alpha) * (level[i] + trend[i]))
        else:
            level.append(alpha * (Y[i] / season[i]) + (1 - alpha) *
                         (level[i] + trend[i]))
        trend.append(beta * (level[i + 1] - level[i]) + (1 - beta) * trend[i])
        season.append(gamma * (Y[i] / (level[i] + trend[i])) +
                      (1 - gamma) * season[i])
        forecast.append((level[i + 1] + trend[i + 1]) * season[i + 1])

    return forecast