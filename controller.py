import model
import view


class Controller:

    def __init__(self, view, model):

        self.view = view
        self.model = model
        self.connect_signals()


    def connect_signals(self):
        """Connect signals and slots."""
        self.view.import_button.clicked.connect(self.import_file)

    def import_file(self):
        self.view.show_directory()
        self.model.import_report(self.view.explorer)

