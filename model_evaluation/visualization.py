import matplotlib.pyplot as plt

def plot_training_history(dates, y_test, predictions):
    


    plt.figure(figsize=(12,6))
    plt.plot(dates, y_test, label= 'Actual')
    plt.plot(dates, predictions, label='Predicted')
    plt.legend()
    plt.title('Demand Forcast vs Actual')
    plt.xlabel('Date')
    plt.ylabel('Quantity')

    plt.tight_layout()
    plt.close()