from math import sqrt


def rmseHoltWintersMultiplicative(params, *args):
    Y = args[0]
    seasonPeriod = args[1]

    trainForecastLen = len(Y) - seasonPeriod
    yTrain = Y[0:trainForecastLen]

    alpha, beta, gamma = params
    level = [sum(yTrain[0:seasonPeriod]) / float(seasonPeriod)]
    trend = [(sum(yTrain[seasonPeriod:2 * seasonPeriod]) -
              sum(Y[0:seasonPeriod])) / seasonPeriod ** 2]
    season = [Y[i] / (level[0] + trend[0])for i in range(seasonPeriod)]
    forecast = [(level[0] + trend[0]) * season[0]]

    for i in range(len(yTrain) + seasonPeriod):
        if i == len(yTrain):
            yTrain.append((level[-1] + trend[-1]) * season[-seasonPeriod])

        if(season[i] < 10 ** 8):
            level.append(alpha * (yTrain[i]) + (1 - alpha) *
                         (level[i] + trend[i]))
        else:
            level.append(alpha * (yTrain[i] / season[i]) + (1 - alpha) *
                         (level[i] + trend[i]))

        trend.append(beta * (level[i + 1] - level[i]) + (1 - beta) * trend[i])
        season.append(gamma * (yTrain[i] / (level[i] + trend[i])) +
                      (1 - gamma) * season[i])
        forecast.append((level[i + 1] + trend[i + 1]) * season[i + 1])

    rmse = sqrt(sum([(k - n) ** 2 for k,
                n in zip(Y[:], forecast[:])]) / len(Y))
    return rmse
