import dearpygui.dearpygui as dpg
from tkinter import Tk, filedialog
import joblib
import numpy as np
import pandas as pd
import data_processing
import time


def Model(file_path_for_fit, file_path_for_predicate, method_num, n_clusters):
    with dpg.window(label="Model processing", width=500, height=200, pos=(50, 100), tag="wait_window",
                    no_resize=True, no_close=True):
        dpg.add_text(label="Please wait for predict", show_label=True, show=True, pos=(-170, 100))
        dpg.add_text(label="Data is now loading...", show_label=True, show=True, pos=(-170, 120), tag="progress_text")
        dpg.add_progress_bar(tag="prediction_progress", default_value=0.0)
    if file_path_for_predicate[-4:] == "xlsx":
        fit_dataframe = pd.read_excel(file_path_for_predicate)
        dpg.set_value("prediction_progress", 0.1)
    elif file_path_for_predicate[-4:] == ".csv":
        fit_dataframe = pd.read_csv(file_path_for_predicate)
        dpg.set_value("prediction_progress", 0.1)
    dpg.set_item_label("progress_text", "Data preprocessing...")
    fit_dataframe = data_processing.process_data(fit_dataframe)
    time.sleep(1)
    dpg.set_value("prediction_progress", 0.2)
    fit_dataframe = data_processing.data_imputer(fit_dataframe)
    time.sleep(1)
    dpg.set_value("prediction_progress", 0.3)
    fit_dataframe = data_processing.normalisation(fit_dataframe)
    time.sleep(1)
    dpg.set_value("prediction_progress", 0.4)
    time.sleep(1)
    dpg.set_value("prediction_progress", 0.5)
    dpg.set_item_label("progress_text", "Fit data...")
    if not dpg.get_value("data_file_path_text"):
        if method_num == "0":
            model = data_processing.KMeans_method(n_clusters)
        elif method_num == "1":
            model = data_processing.AgglomerativeClustering_method(n_clusters)
        elif method_num == "2":
            model = data_processing.DBSCAN_method()
        elif method_num == "3":
            model = data_processing.SpectralClustering_method(n_clusters)
        elif method_num == "4":
            model = data_processing.GaussianMixture_method(n_clusters)
        elif method_num == "5":
            model = data_processing.MeanShift_method()
    else:
        model = joblib.load(file_path_for_fit)
    dpg.set_item_label("progress_text", "Data prediction")
    dpg.set_value("prediction_progress", 0.8)
    predicated_data = data_processing.predict_data(model, fit_dataframe)
    dpg.set_value("prediction_progress", 1.0)
    silhouette_scr = data_processing.sil_plot(fit_dataframe, predicated_data)
    dpg.delete_item("wait_window")
    with dpg.window(tag="test_window", label="Test_window", width=650, height=650,
                    pos=(100, 100), no_resize=True, no_close=True):
        width, height, channels, data = dpg.load_image("Results/result.png")
        with dpg.texture_registry():
            texture_id = dpg.add_static_texture(width, height, data)
        dpg.add_text("Prediction finished!")
        dpg.add_text("Silhoette figure:")
        dpg.add_image(texture_id)
        dpg.add_text("Silhoette score = " + str(silhouette_scr))
        with dpg.group(horizontal=True):
            dpg.add_button(label="Return", callback=returner)
            dpg.add_button(label="Save", callback=save_(fit_dataframe, predicated_data, model))


def save_(data, fitted, model):
    data.insert(loc=len(data.columns), column='cluster', value=fitted)
    data.to_csv("Results/result.csv")
    joblib.dump(model, "Results/result.joblib")
    print("done")


def returner():
    dpg.delete_item("test_window")


def on_select_fit_pressed(widget):
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("JobLib file", "*.joblib")])
    if file_path:
        dpg.set_value("data_file_path_text", file_path)


def on_select_pred_pressed(widget):
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx"), ("Excel files", "*.csv")])
    if file_path:
        dpg.set_value("pred_file_path_text", file_path)


def help_item(widget):
    with dpg.window(label="Help", tag="help_window"):
        dpg.add_text("How to use:")
        dpg.add_text("1) Select clusterisation method from node;")
        dpg.add_text("1.1) If you are going to use a file, then the choice of method is not required;")
        dpg.add_text("2) Select .joblib file (if necessary) and .xlsx or .csv file with dataset for predicate;")
        dpg.add_text("3) Enter the required number of clusters in the text field;")
        dpg.add_text("4) Press start button and wait;")
        dpg.add_text("5) On final window you can inspect sillhouette analyze results and save data")
        dpg.add_text(" (all data saves to Results/) or return on main window and repeat")

def method_selector(widget):
    if dpg.get_value("select_method_radio"):
        if dpg.get_value("select_method_radio") == 'K-Means':
            dpg.set_value("sel", "0")
        if dpg.get_value("select_method_radio") == 'Agglomerative clustering':
            dpg.set_value("sel", "1")
        if dpg.get_value("select_method_radio") == 'DBSCAN':
            dpg.set_value("sel", "2")
        if dpg.get_value("select_method_radio") == 'Spectral clustering':
            dpg.set_value("sel", "3")
        if dpg.get_value("select_method_radio") == 'Gaussian mixture':
            dpg.set_value("sel", "4")
        if dpg.get_value("select_method_radio") == 'Mean shift':
            dpg.set_value("sel", "5")


def start_method(widget):
    if dpg.get_value("pred_file_path_text") and dpg.get_value("n_clusters"):
        Model(dpg.get_value("data_file_path_text"), dpg.get_value("pred_file_path_text"), dpg.get_value("sel"),
              int(dpg.get_value("n_clusters")))
    elif not dpg.get_value("pred_file_path_text"):
        dpg.set_item_label("not_file_pred", "Please, select file for predicate")
        dpg.set_item_label("not_n_clusters", "")
    elif not dpg.get_value("n_clusters"):
        dpg.set_item_label("not_n_clusters", "Please, select number of clusters")
        dpg.set_item_label("not_file_pred", "")


def exit():
    dpg.delete_item("Window")


def main_interface():
    dpg.create_context()
    with dpg.font_registry():
        default_font = dpg.add_font("Machine BT.ttf", size=18)
    with dpg.window(label="Clusterisation machine learning model", tag="Window",  width=600, height=800):
        with dpg.menu_bar():
            with dpg.menu(label="File"):
                dpg.add_menu_item(label="Exit", callback=exit)
            with dpg.menu(label="Help"):
                dpg.add_menu_item(label="Help", callback=help_item)
        dpg.add_text(default_value="0", tag="sel", show=False)
        with dpg.tree_node(label="Select clusterisation method:"):
            dpg.add_radio_button(tag="select_method_radio", items=('K-Means', 'Agglomerative clustering', 'DBSCAN',
                                                         'Spectral clustering', 'Gaussian mixture',
                                                                   'Mean shift'), callback=method_selector)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Select .joblib file if have fitted model", callback=on_select_fit_pressed)
            dpg.add_text("", tag="data_file_path_text")
        with dpg.group(horizontal=True):
            dpg.add_button(label="Select .xlsx, .csv file with dataset for predict", callback=on_select_pred_pressed)
            dpg.add_text("", tag="pred_file_path_text")
        with dpg.group(horizontal=True):
            dpg.add_text("Number of clusters (don't using for DBSCAN and Mean Shift method, but required)")
            dpg.add_input_text(tag="n_clusters", default_value="2")
        dpg.add_text(label="", tag="not_file_pred", show_label=True, show=True, pos=(-170, 140))
        dpg.add_text(label="", tag="not_n_clusters", show_label=True, show=True, pos=(-170, 150))
        dpg.add_button(label="Start prediction", callback=start_method)

        dpg.bind_font(default_font)

    dpg.create_viewport(title='ML cluster', width=800, height=800, resizable=False)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("Window", True)
    dpg.start_dearpygui()
    dpg.destroy_context()


main_interface()
