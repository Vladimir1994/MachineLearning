import matplotlib.pyplot as plt
from numpy import random

from predict import predict


def main():
    x = []
    y_additive = []
    y_multiplicative = []

    season = [0.0000, 0.0003, 0.0111, 0.1353, 0.6065, 1.0000, 0.6065, 0.1353,
              0.0111, 0.0003, 0.0000]
    noise = 0 * random.randn(200)

    for i in range(-100, 100):
        x.append(float(i) / 10)
        y_additive.append(1.3 ** x[-1])
        y_multiplicative.append(1.3 ** x[-1])

    for i in range(len(x)):
        y_additive[i] = y_additive[i] + season[i % 10] + noise[i]
        y_multiplicative[i] = y_multiplicative[i] * season[i % 10] + noise[i]

    forecast_1 = predict(y_additive[:-20], 10, 19)
    forecast_2 = predict(y_multiplicative[:-20], 10, 19)
    plt.plot(forecast_1)
    plt.plot(y_additive)
    plt.figure(2)
    plt.plot(y_multiplicative)
    plt.plot(forecast_2)
    plt.show()


if __name__ == "__main__":
    main()