import matplotlib.pyplot as plt
from numpy import random

from predict import predict


def main():
    x = []
    yA = []
    yM = []

    season = [0.0000, 0.0003, 0.0111, 0.1353, 0.6065, 1.0000, 0.6065, 0.1353,
              0.0111, 0.0003, 0.0000]
    noise = 0 * random.randn(200)

    for i in range(-100, 100):
        x.append(float(i) / 10)
        yA.append(1.3 ** x[-1])
        yM.append(1.3 ** x[-1])

    for i in range(len(x)):
        yA[i] = yA[i] + season[i % 10] + noise[i]
        yM[i] = yM[i] * season[i % 10] + noise[i]

    forecast_1 = predict(yA[:-20], 10, 19)
    forecast_2 = predict(yM[:-20], 10, 19)
    plt.plot(forecast_1)
    plt.plot(yA)
    plt.figure(2)
    plt.plot(yM)
    plt.plot(forecast_2)
    plt.show()


if __name__ == "__main__":
    main()