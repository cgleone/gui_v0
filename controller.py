import model
import view


class Controller:

    def __init__(self, view, model):

        self.view = view
        self.model = model
        self.create_table_grid()
        self.create_settings_dialog()
        self.view.populate_dialog(self.model.display_names)
        self.connect_signals()


        # when we go to a patient's report screen:
        self.initial_reports_display()



    def connect_signals(self):
        """Connect signals and slots."""
        self.view.import_button.clicked.connect(self.import_file)
        self.view.filter_button.clicked.connect(self.filter_selection)
        self.view.dialog_button.clicked.connect(self.apply_filters)
        self.view.report_table.cellPressed.connect(self.view_report)
        self.view.go_button.clicked.connect(self.begin_search)
        self.view.search_bar.returnPressed.connect(self.begin_search)
        self.view.clear_filters_button.clicked.connect(self.clear_filters)
        self.view.dialog_clear_filters_button.clicked.connect(self.clear_dialog_filters)
        self.view.remove_filter_buttons.buttonClicked.connect(self.remove_one_filter)
        self.view.settings_button.clicked.connect(self.select_settings)
        self.view.apply_settings_button.clicked.connect(self.apply_settings)
        self.view.clear_display_name_group.buttonClicked.connect(self.reset_single_display_name)
        self.view.reset_display_names.clicked.connect(self.reset_all_display_names)


    def view_report(self, row, col):
        filename, isPDF, name = self.model.view_report(row, col)
        if isPDF:
            self.view.display_pdf(filename, name, row, col)
        else:
             self.view.display_image_report(filename, name)

    def begin_search(self):
        user_query = self.view.search_bar.text()
        report_IDs = self.model.search(user_query)
        self.get_report_info_to_display(report_IDs)
        self.display_report_info()


    def import_file(self):
        self.view.show_directory()
        if self.view.explorer:
            self.file_selection = self.view.explorer
            self.view.import_enabled(False)
            # start a thread
            import_file_thread = self.view.create_thread(self)
            import_file_thread.finished.connect(self.display_report_info)
            import_file_thread.start()

    def thread_interior(self):
        self.model.import_report(self.file_selection)
        self.get_report_info_to_display()
       # self.view.worker.worker_finished()

    def filter_selection(self):
        self.view.show_dialog()

    def apply_filters(self):
        active_filters = self.model.set_filters(self.view.mod_options, self.view.bodypart_options, self.view.date_options)
        self.populate_filter_layout(active_filters)
        self.get_filtered_reports()
        self.view.close_dialog()

    def populate_filter_layout(self, active_filters):
        self.model.clear_layout(self.view.filters_layout)
        if len(active_filters) != 0:
            self.view.populate_filters_layout(active_filters)

    def get_filtered_reports(self):
        report_IDs = self.model.get_filtered_ids()
        self.get_report_info_to_display(report_IDs)
        self.display_report_info()

    def clear_filters(self):
        self.model.clear_layout(self.view.filters_layout)
        self.model.reset_current_filters()
        self.model.reset_filter_checkboxes([self.view.mod_options, self.view.bodypart_options, self.view.date_options])
        self.initial_reports_display()

    def clear_dialog_filters(self):
        self.model.reset_filter_checkboxes([self.view.mod_options, self.view.bodypart_options, self.view.date_options])

    def remove_one_filter(self, filter_to_remove):
        self.model.uncheck_filter(filter_to_remove, [self.view.mod_options, self.view.bodypart_options, self.view.date_options])
        self.apply_filters()

    def initial_reports_display(self):
        self.get_report_info_to_display()
        self.display_report_info()

    def get_report_info_to_display(self, ids=None):
        self.reports = self.model.get_reports_to_display(ids)
        self.rows = len(self.reports)


    def display_report_info(self):
        self.view.set_table_row_count(self.rows)
        self.view.populate_report_table(self.reports)
        self.view.import_enabled(True)

    def create_table_grid(self):
        current_categories = self.model.current_categories
        self.view.create_table_grid(current_categories)

    def select_settings(self):
        self.view.show_settings_dialog()

    def apply_settings(self):
        # determine which checkboxes are checked
        self.model.determine_checked_categories(self.view.category_list)
        self.view.create_table_columns(self.model.current_categories)
        # determine new display names
        self.model.determine_display_names(self.view.display_names_table)
        self.view.populate_display_names_table(self.model.display_names)
        self.get_report_info_to_display(self.model.current_report_IDs)
        self.display_report_info()
        self.model.update_filter_ceckmark_display_text([self.view.mod_options, self.view.bodypart_options])
        self.view.close_settings_dialog()

    def create_settings_dialog(self):
        self.view.create_categories()
        self.model.update_view_category_list(self.view.category_list)
        self.view.create_settings_dialog_for_later(self.model.display_names)

    def reset_single_display_name(self, button_pressed):
        self.model.reset_single_display_name(button_pressed, self.view.clear_display_name_group, self.view.display_names_table)

    def reset_all_display_names(self):
        self.model.reset_all_display_names(self.view.display_names_table)