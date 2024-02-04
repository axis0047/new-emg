import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer
import pyqtgraph as pg

class MyMainWindow(QMainWindow):
    def __init__(self):
        super(MyMainWindow, self).__init__()

        # Set up the main window
        self.setWindowTitle('EMG Data')
        self.setGeometry(100, 100, 800, 600)

        # Create the central widget and the layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create a pyqtgraph PlotWidget
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)

        # Create a simple plot
        self.plot()

        # Set up a timer to update the plot every second
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(1000)  # Update every 1000 milliseconds (1 second)

    def plot(self):
        # Plot some example data
        self.x = []
        self.y = []

        # Plot the data on the PlotWidget
        self.plot_curve = self.plot_widget.plot(self.x, self.y, name='EMG Data', symbol='o', symbolSize=10)

    def update_plot(self):
        # Dummy data for testing
        dummy_data = [1, 3, 5, 2, 4]

        # Call the update function with the dummy data
        self.update_plot_with_data(dummy_data)

    def update_plot_with_data(self, data):
        # Update the data
        self.x = list(range(1, len(data) + 1))
        self.y = data

        # Update the plot data
        self.plot_curve.setData(self.x, self.y)

def plotter(data_array):
        # Your custom function that generates data and updates the plot
    window = MyMainWindow()

    # Generate your custom data
    # Call the update_plot_with_data with the custom data
    window.update_plot_with_data(data_array)

    # Show the window and start the application event loop
    window.show()

