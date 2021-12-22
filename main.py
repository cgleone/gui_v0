from view import View
from controller import Controller
from model import Model
import sys
from PyQt5.QtWidgets import QApplication

__version__ = '0.1'
__author__ = 'Tigger and Cathleen'




def launch():
    """Main function."""
    # Create an instance of QApplication
    app = QApplication(sys.argv)
    view = View()
    model = Model()
    controller = Controller(view, model)

    view.show()

    # Execute the app's main loop
    sys.exit(app.exec_())

if __name__ == '__main__':
    launch()



