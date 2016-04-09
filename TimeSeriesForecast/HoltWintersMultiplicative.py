from math import sqrt

from scipy.optimize import minimize
from numpy import array


def holt_winters_multiplicative(x, season_period, forecast_len):
    y = x[:]
    init_values = array([0.5, 0.5, 0.5])
    boundaries = [(0, 1), (0, 1), (0, 1)]

    parameters = minimize(rmse_holt_winters_multiplicative, x0=init_values,
                          args=(y, season_period), bounds=boundaries)
    alpha, beta, gamma = parameters.x

    level = [sum(y[0:season_period]) / float(season_period)]
    trend = [(sum(y[season_period:2 * season_period])
              - sum(y[0:season_period]))
             / season_period ** 2]
    season = [y[i] / (level[0] + trend[0]) for i in range(season_period)]
    forecast = [(level[0] + trend[0]) * season[0]]

    for i in range(len(y) + forecast_len):
        if i == len(y):
            y.append((level[-1] + trend[-1]) * season[-season_period])

        if season[i] < 10 ** 8:
            level.append(alpha * (y[i]) + (1 - alpha) * (level[i] + trend[i]))
        else:
            level.append(alpha * (y[i] / season[i]) + (1 - alpha) *
                         (level[i] + trend[i]))
        trend.append(beta * (level[i + 1] - level[i]) + (1 - beta) * trend[i])
        season.append(gamma * (y[i] / (level[i] + trend[i])) +
                      (1 - gamma) * season[i])
        forecast.append((level[i + 1] + trend[i + 1]) * season[i + 1])

    return forecast


def rmse_holt_winters_multiplicative(params, *args):
    Y = args[0]
    season_period = args[1]

    train_forecast_len = len(Y) - season_period
    y_train = Y[0:train_forecast_len]

    alpha, beta, gamma = params
    level = [sum(y_train[0:season_period]) / float(season_period)]
    trend = [(sum(y_train[season_period:2 * season_period]) -
              sum(Y[0:season_period])) / season_period ** 2]
    season = [Y[i] / (level[0] + trend[0])for i in range(season_period)]
    forecast = [(level[0] + trend[0]) * season[0]]

    for i in range(len(y_train) + season_period):
        if i == len(y_train):
            y_train.append((level[-1] + trend[-1]) * season[-season_period])

        if season[i] < 10 ** 8:
            level.append(alpha * (y_train[i]) + (1 - alpha) *
                         (level[i] + trend[i]))
        else:
            level.append(alpha * (y_train[i] / season[i]) + (1 - alpha) *
                         (level[i] + trend[i]))

        trend.append(beta * (level[i + 1] - level[i]) + (1 - beta) * trend[i])
        season.append(gamma * (y_train[i] / (level[i] + trend[i])) +
                      (1 - gamma) * season[i])
        forecast.append((level[i + 1] + trend[i + 1]) * season[i + 1])

    rmse = sqrt(sum([(k - n) ** 2 for k,
                n in zip(Y[:], forecast[:])]) / len(Y))
    return rmse
