import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_results(training_stats, train_loss_set):
    # Display floats with two decimal places.
    pd.set_option("precision", 2)
    # Create a DataFrame from our training statistics.
    df_stats = pd.DataFrame(data=training_stats)
    # Use the 'epoch' as the row index.
    df_stats = df_stats.set_index("epoch")

    print(df_stats)
    # Use plot styling from seaborn.
    sns.set(style="darkgrid")

    # Increase the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12, 6)

    # Plot the learning curve.
    plt.plot(df_stats["Training Loss"], "b-o", label="Training")
    plt.plot(df_stats["Valid. Loss"], "g-o", label="Validation")

    # Label the plot.
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks([1, 2, 3, 4])
    plt.show()

    # Plot the training loss by batch
    plt.title("Training loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.plot(train_loss_set)
    plt.show()
