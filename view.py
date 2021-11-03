import controller

import sys

# Import QApplication and the required widgets from PyQt5.QtWidgets
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QWidget

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGridLayout
from PyQt5.QtWidgets import QLineEdit
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QVBoxLayout

class MainWindow(QMainWindow):
    """Test App View (GUI)."""
    def __init__(self):
        """View initializer."""
        super().__init__()


def launch():
    """Main function."""
    # Create an instance of QApplication
    app = QApplication(sys.argv)
    view = MainWindow()
    view.show()
    # Execute the app's main loop
    sys.exit(app.exec_())

