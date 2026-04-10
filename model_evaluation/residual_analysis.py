import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from pathlib import Path


def plot_residuals(y_test, predictions):

    Path('output').mkdir(exist_ok=True)

    residuals = y_test - predictions
    plt.figure(figsize=(10,5))
    plt.hist(residuals, bins=50)
    plt.title("Residuals Distribution")
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('output/residuals_distribution.png')
    plt.close()

    plt.figure(figsize=(10,5))

    plot_acf(residuals, lags=30)
    plt.title('Residual Autocrrelation')
    plt.tight_layout()
    plt.savefig('output/residuals_acf.png')
    plt.close()

    return residuals