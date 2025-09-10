'''
###
## NLSA GUI 
## Version 1.0
## UWM, July 31, 2025
'''
import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QGridLayout, QWidget,
    QLabel, QPushButton, QLineEdit, QComboBox, QGroupBox, QStatusBar, QMessageBox, QFileDialog, QSpacerItem,
    QSizePolicy, QFrame, QProgressBar
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import os
import h5py
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys
import numpy as np
import os
import time
from matplotlib.patches import Rectangle

cxfel_root = os.environ['CXFEL_ROOT']
startup_file = cxfel_root + '/misc_tools/startup.py'
exec(open(startup_file).read())


class Stream(QObject):
    newtext = pyqtSignal(str)

    def write(self, text):
        self.newtext.emit(str(text))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive Manifold-based Data Analysis Tool")
        self.setGeometry(200, 100, 1200, 800)

        self.tabs = QTabWidget(self, tabShape=QTabWidget.Rounded)
        self.tabs.tabCloseRequested.connect(self.close_current_tab)
        self.tabs.setTabsClosable(True)

        self.tab1 = QWidget()
        self.tabs.addTab(self.tab1, "Main Window")

        self.selected_file = ""
        self.num_snapshots = 1000
        self.nn = [50, 100]
        self.sigfac = [2, 5]
        self.variable_name = ""
        self.v_list = []
        self.v_name = ""
        self.concat_num = 1
        self.n = 500
        self.mpi = 4
        self.maxnn = 10
        self.minnn = 10
        self.N = 10
        self.D = 1
        self.completed_percent = 1
        self.file_path = " "
        self.selected_indices = set()  # Set to store multiple selected indices
        self.red_boxes = []  # Holds the red border rectangles for each subplot
        self.axes = []       # Holds references to the subplot axes
        self.num_copy = 2
        self.nlsa_file = ""

        # Central widget
        self.setCentralWidget(self.tabs)

        # Main layout
        main_layout = QHBoxLayout(self.tab1)

        # splitting the windows here
        splitter = QSplitter()

        # Part 1: Parameter Search Inputs
        self.param_layout_widget = QWidget()
        self.param_layout = QVBoxLayout(self.param_layout_widget)
        self.create_parameter_search_layout(self.param_layout)
        self.param_layout_widget.setMinimumWidth(300)
        splitter.addWidget(self.param_layout_widget)

        # Part 2: Interactive Heatmap
        self.heatmap_layout_widget = QWidget()
        self.heatmap_layout = QVBoxLayout(self.heatmap_layout_widget)
        self.create_heatmap_layout(self.heatmap_layout)
        splitter.addWidget(self.heatmap_layout_widget)

        # Set splitter sizes (30:70 ratio)
        splitter.setSizes([300, 700])

        # Add splitter to main layout
        main_layout.addWidget(splitter)

        # Status bar for feedback
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Set background color
        self.setStyleSheet("background-color: #d3d3d3;")  # Gray background

    def close_current_tab(self, i):
        
        self.tabs.removeTab(i)
        #if there is only one tab
        if i==0:
            super(QWidget).close()

    def closeEvent(self, event):
        """Shuts down application on close."""
        # Return stdout to defaults.
        sys.stdout = sys.__stdout__
        super().closeEvent(event)

    def select_file(self):
        """Open a file dialog to select a data file."""
        try:
            self.file_path, _ = QFileDialog.getOpenFileName(self, "Select Data File", './',
                                                            "H5 Files (*.h5);;All Files (*)")
            self.load_message()
        except Exception as e:
            print(e)

    def load_message(self):
        if self.file_path:
            import h5py
            f = h5py.File(self.file_path, 'r')
            self.v_list = list(f.keys())
            self.status_bar.showMessage(f"Selected file: {self.file_path}")
            if(self.variable_name):
                self.variable_name.clear()
            self.variable_name.addItems(self.v_list)
            if self.v_list:  # Ensure list is not empty
                self.v_name = self.v_list[0]  # Select the first dataset by default
            else:
                self.v_name = None
            self.v_name = self.variable_name.currentText()
            if self.v_name:  # Check if a valid variable is selected
                self.xyz = f[self.v_name]
                self.D = int(self.xyz.shape[1])  # Number of Pixels
                self.N = int(self.xyz.shape[0])
                self.suggest_nn = [int(self.N * 0.05), int(self.N * 0.10), int(self.N * 0.15), int(self.N * 0.20)]
                self.data_info.setText(f'Number of Pixels: {self.D}\nNumber of Snapshots: {self.N}\n'
                                    f'Typical nN values: {self.suggest_nn[0]} {self.suggest_nn[1]} {self.suggest_nn[2]} {self.suggest_nn[3]} \n'
                                    #f'Typical C values : {int(self.N * 0.10)}')
                                    f'Typical C values: {int(self.N * 0.05)} to {int(self.N * 0.10)}')
 
    # saving data and showing message to the user
    def save_data(self):
        try:
            self.v_name = self.variable_name.currentText()
            self.nn = list(map(int, self.nn_list.text().split()))
            self.sigfac = list(map(int, self.sigma_value.text().split()))
            self.nEigs = int(self.n_eigens.text())
            self.concat_num = int(self.c_values.text())
            self.n = int(self.chunk_size.text())
            self.mpi = int(self.mpi_workers.text())
            # Define file path for storing the .h5 file
            save_dir = "gui_params"  # Directory to store runs
            os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
            save_file = os.path.join(save_dir, "gui_parameters.h5")
            # Save the extracted data in an HDF5 file
            with h5py.File(save_file, 'a') as f:  # Open in append mode
                run_id = f"run_{len(f.keys())+1}"  # Unique run ID
                grp = f.create_group(run_id)  # Create a new group for this run
                # Store data as datasets inside the run group
                grp.create_dataset("file_path", data=self.file_path)
                grp.create_dataset("variable_name", data=self.v_name)
                grp.create_dataset("nN", data=self.nn)
                grp.create_dataset("sigma_factor", data=self.sigfac)
                grp.create_dataset("nEigs", data=self.nEigs)
                grp.create_dataset("concat_num", data=self.concat_num)
                grp.create_dataset("chunk_size", data=self.n)
                grp.create_dataset("mpi_workers", data=self.mpi)

            text = " Data = {} \n Variable Name = {} \n  nN = {} \n Sigma Factor = {} \n nEigs = {}".format(
                self.file_path, self.v_name, self.nn, self.sigfac, self.nEigs)
            self.status_bar.showMessage(text)
        except ValueError as e:
            QMessageBox.warning(self, "Input Error", str(e))
            return

    def create_parameter_search_layout(self, main_layout):
        """Create the layout for parameter search inputs."""
        param_layout = QVBoxLayout()

        # Group box for search parameters
        param_group = QGroupBox("Settings")
        self.p_layout = QGridLayout()

        # File selection button
        self.select_file_button = QPushButton("Select Data File")
        self.select_file_button.clicked.connect(self.select_file)
        self.p_layout.addWidget(self.select_file_button, 0, 0, 1, 2)

        # Variable names
        self.variable_name = QComboBox()
        self.p_layout.addWidget(QLabel("Data Matrix:"), 1, 0)
        self.p_layout.addWidget(self.variable_name, 1, 1)

        # C Values dropdown
        self.c_values = QLineEdit()
        self.c_values.setPlaceholderText("Ex: 10 (C>=2) ")
        self.p_layout.addWidget(QLabel("Concatenation Value (C):"), 2, 0)
        self.p_layout.addWidget(self.c_values, 2, 1)

        # List of NN
        self.nn_list = QLineEdit()
        self.nn_list.setPlaceholderText("Ex: 100 200 400")
        self.p_layout.addWidget(QLabel("Nearest Neighbors (nN):"), 3, 0)
        self.p_layout.addWidget(self.nn_list, 3, 1)

        # Sigma Value
        self.sigma_value = QLineEdit()
        self.sigma_value.setPlaceholderText("Ex: 2 4 8")
        self.p_layout.addWidget(QLabel("Sigma Factors:"), 4, 0)
        self.p_layout.addWidget(self.sigma_value, 4, 1)

        # N Eigs
        self.n_eigens = QLineEdit()
        self.n_eigens.setPlaceholderText("Ex: 5")
        self.p_layout.addWidget(QLabel("Number of Eigenvalues:"), 5, 0)
        self.p_layout.addWidget(self.n_eigens, 5, 1)

        # chunk size
        self.chunk_size = QLineEdit()
        self.chunk_size.setPlaceholderText("Ex: 500")
        self.p_layout.addWidget(QLabel("Data Chunk Size:"), 6, 0)
        self.p_layout.addWidget(self.chunk_size, 6, 1)

        # mpi workers
        self.mpi_workers = QLineEdit()
        self.mpi_workers.setPlaceholderText("Ex: 4")
        self.p_layout.addWidget(QLabel("Number of MPI Workers:"), 7, 0)
        self.p_layout.addWidget(self.mpi_workers, 7, 1)

        # Run button
        self.pm_button = QPushButton("Run Parameter Search")
        self.pm_button.clicked.connect(self.run_analysis)
        self.p_layout.addWidget(self.pm_button, 8, 0, 1, 2)

        # Choose nN and sigma
        self.nN_chosen = QLineEdit()
        self.nN_chosen.setPlaceholderText("")
        self.p_layout.addWidget(QLabel("Selected nN"), 9, 0)
        self.p_layout.addWidget(self.nN_chosen, 9, 1)

        #  Selected Sigma Factor
        self.sigma_chosen = QLineEdit()
        self.sigma_chosen.setPlaceholderText("")
        self.p_layout.addWidget(QLabel("Selected Sigma Factor"), 10, 0)
        self.p_layout.addWidget(self.sigma_chosen, 10, 1)

        # Plot DM Eigenfunctions
        self.dm_plot_button = QPushButton("Plot DM Eigenfunctions")
        self.dm_plot_button.clicked.connect(self.eig_plot)
        self.p_layout.addWidget(self.dm_plot_button, 11, 0, 1, 2)
        
        #Choose Number of copies
        self.num_copy_field = QLineEdit()
        self.num_copy_field.setPlaceholderText("Ex: 2 (nCopy<C)")
        self.p_layout.addWidget(QLabel("Number of copies (nCopy)"), 12, 0)
        self.p_layout.addWidget(self.num_copy_field, 12, 1)
        
        # NLSA Button
        self.nlsa_button = QPushButton("Run NLSA")
        self.nlsa_button.clicked.connect(self.run_nlsa)
        self.p_layout.addWidget(self.nlsa_button, 13, 0)

        # NLSA Reconstruct Button
        self.nlsa_button = QPushButton("Run NLSA Reconstruction")
        self.nlsa_button.clicked.connect(self.reconstruct)
        self.p_layout.addWidget(self.nlsa_button, 13, 1)


        # Selected Chronos
        self.chronos_selected = QLabel(" ")
        self.chronos_selected.setWordWrap(True)  # Allow text to wrap if too long
        self.chronos_selected.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.p_layout.addWidget(self.chronos_selected, 14, 0, 1, 2)

        # Chronos file confirmation msg
        self.chronos_file_msg = QLabel(" ")
        self.chronos_file_msg.setWordWrap(True)  # Allow text to wrap if too long
        self.chronos_file_msg.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.p_layout.addWidget(self.chronos_file_msg, 15, 0, 1, 2)

        # Data info Label
        self.data_info = QLabel(" ")
        self.data_info.setWordWrap(True)  # Allow text to wrap if too long
        self.data_info.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.p_layout.addWidget(self.data_info, 16, 0, 3, 2)

        # Progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.p_layout.addWidget(self.progress_bar, 19, 0, 2, 4)
        self.progress_bar.hide()

        self.p_layout.setSpacing(20)  # Reduce spacing between rows

        param_group.setLayout(self.p_layout)
        param_layout.addWidget(param_group)
        main_layout.addLayout(param_layout)

    
    def create_heatmap_layout(self, main_layout):
        """Create the layout for the interactive heatmap."""
        heatmap_layout = QVBoxLayout()

        # Group box for heatmap
        heatmap_group = QGroupBox("Interactive Heatmap")
        heatmap_box_layout = QVBoxLayout()

        self.heatmap_canvas = FigureCanvas(Figure())
        self.heatmap_canvas.mpl_connect("button_press_event", self.on_heatmap_click)
        heatmap_box_layout.addWidget(self.heatmap_canvas)

        heatmap_group.setLayout(heatmap_box_layout)
        heatmap_layout.addWidget(heatmap_group)
        main_layout.addLayout(heatmap_layout)


    def run_analysis(self):
        """Collect parameters and run the analysis."""
        if not self.file_path or not os.path.exists(self.file_path):
            raise ValueError("No valid file selected. Please select a valid data file.")
        self.status_bar.showMessage("Loading data...")
        try:
            # Prepare the correlation data
            self.progress_bar.show()
            self.save_data()
            self.completed_percent = 1
            if hasattr(self, 'selected_rect'):
                self.selected_rect.remove()
                del self.selected_rect
            self.progress_bar.setValue(self.completed_percent)
            data = self.click_run()
            if data is None:
                return  # Error already handled in compute_correlations
            # Update the heatmap
            self.update_heatmap(data)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An unexpected error occurred: {str(e)}")

    def corr_no_noisy(self, a):
        num_eig = a.shape[1]
        non_noise_list = []
        for i in range(1, num_eig):
            if np.max(a[:10, i]) - np.min(a[:10, i]) < 9e-4:
                non_noise_list.append(i)
        return non_noise_list

    def eig_plot(self):
        nN_1 = int(self.nN_chosen.text())
        sigma_1 = float(self.sigma_chosen.text())
        eigVec_list = h5py.File('eigVec_list.h5')['eigVec']
        eigVal_list = h5py.File('eigVal_list.h5')['eigVal']
        nN_1_index = self.nn.index(nN_1)
        sigma_1_index = self.sigfac.index(sigma_1)
        self.eigVec = eigVec_list[nN_1_index][sigma_1_index]
        eigVal = eigVal_list[nN_1_index][sigma_1_index]
        for i in range(self.nEigs):
            ax0 = plt.subplot2grid((self.nEigs, 1), (i, 0))
            ax0.plot(self.eigVec[:, i + 1] / self.eigVec[:, 0])
            ax0.set_ylabel(r'$\psi_{}$'.format(i + 1))
            if i == 0:
                ax0.set_title(r'Eigenfunctions for nN={}, $\sigma_{{factor}}$={}'.format(nN_1, sigma_1))
            if i < (self.nEigs - 1):
                ax0.set_xticks([])
            else:
                ax0.set_xlabel('Time Delay')
        plt.tight_layout()
        plt.savefig('DM_eigVecs.png')
        plt.close()

        plt.scatter(np.arange(self.nEigs), eigVal[1:], s=60)
        plt.plot(np.arange(self.nEigs), eigVal[1:], linestyle='dashed')
        plt.xlabel('Eigenvalue Index')
        plt.ylabel('Eigenvalue')
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.title(r'Eigenvalues for nN={}, $\sigma_{{factor}}$={}'.format(nN_1, sigma_1))
        plt.tight_layout()
        plt.savefig('DM_eigVals.svg')
        plt.close()

        self.nN_2 = nN_1
        self.sigma_2 = sigma_1

        self.tab2 = QWidget()
        tab2_layout = QHBoxLayout(self.tab2)
        label1 = FigureCanvas(Figure())
        label1.figure.clear()
        grid = label1.figure.add_gridspec(self.nEigs, 1, hspace=0.15)
        for i in range(self.nEigs):
            ax0 = label1.figure.add_subplot(grid[i, 0])
            ax0.plot(self.eigVec[:, i + 1] / self.eigVec[:, 0])
            ax0.set_ylabel(r'$\psi_{}$'.format(i + 1), fontsize=16)
            if i == 0:
                ax0.set_title(r'Eigenfunctions for nN={}, $\sigma_{{factor}}$={}'.format(nN_1, sigma_1))
            if i < (self.nEigs - 1):
                ax0.set_xticks([])
                ax0.tick_params(axis='y', labelsize=14)
            else:
                ax0.set_xlabel('Time Delay', fontsize=16)
                ax0.tick_params(axis='both', labelsize=14)
            
        label1.figure.tight_layout()
        label1.figure.subplots_adjust(top=0.95, bottom=0.08, left=0.15, right=0.95)  # Adjust top margin
        label1.draw()

        label2 = FigureCanvas(Figure())
        label2.figure.clear()
        ax = label2.figure.add_subplot()
        ax.plot(np.arange(self.nEigs), eigVal[1:], linestyle='dashed')
        ax.scatter(np.arange(self.nEigs), eigVal[1:], s=60)
        ax.set_xlabel('Eigenvalue Index', fontsize=16)
        ax.set_ylabel('Eigenvalue', fontsize=16)
        ax.tick_params(axis='both', labelsize=14)
        ### Shift tick labels by +1
        ticks_ = np.arange(len(np.arange(self.nEigs)))         # [0, 1, 2, 3, 4]
        labels_ = [str(ii + 1) for ii in ticks_]  # ['1', '2', '3', '4', '5']
        ax.set_xticks(ticks_)
        ax.set_xticklabels(labels_)
        ax.set_title(r'Eigenvalues for nN={}, $\sigma_{{factor}}$={}'.format(nN_1, sigma_1))
        label2.figure.tight_layout()
        label2.draw()

        # label.show()
        tab2_layout.addWidget(label1)
        tab2_layout.addWidget(label2)
        self.tabs.addTab(self.tab2, "DM of nN="+self.nN_chosen.text()+",\u03C3="+self.sigma_chosen.text())
    
    def is_integer_input(self,input_string):
        try:
            int(input_string)
            return True
        except ValueError:
            return False

    def run_nlsa(self):
        import subprocess
        from misc_tools import write_h5
        import time

        self.chronos_selected.clear()
        self.chronos_file_msg.clear()
        
        
        nN_1 = int(self.nN_chosen.text())
        sigma_1 = float(self.sigma_chosen.text())
        eigVec_list = h5py.File('eigVec_list.h5')['eigVec']
        nN_1_index = self.nn.index(nN_1)
        sigma_1_index = self.sigfac.index(sigma_1)
        self.eigVec = eigVec_list[nN_1_index][sigma_1_index]
        self.nN_2 = nN_1
        self.sigma_2 = sigma_1

        t0 = time.time()
        eigVec = self.eigVec
        mu = (eigVec[:, 0]) * (eigVec[:, 0])
        psi = eigVec[:, 1:].T / eigVec[:, 0]
        write_h5("mu_psi.h5", mu, 'mu')
        write_h5("mu_psi.h5", psi, 'psi')

        num_worker = self.mpi
        data_file = 'data_file_for_sna.h5'
        variable_name = self.v_name
        xyz = self.xyz
        N, D = xyz.shape
        c = self.concat_num
        n = self.n
        nN = self.nn[0]
        if (self.is_integer_input(self.num_copy_field.text())):
            self.num_copy = int(self.num_copy_field.text())
            
        
        if self.num_copy <=0 or self.num_copy >=c:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle('numCopy info')
            msg.setStandardButtons(QMessageBox.Close) 
            msg.setText("nCopy {} is not allowed ".format(self.num_copy))
            msg.exec_()
        else:
            self.progress_bar.show()
            self.completed_percent = 1
            self.progress_bar.setValue(self.completed_percent)
            dotp_code = cxfel_root + "/sna/run_sna_.py"
            dotp_file = 'xtx.h5'  # Place holder, never generated

            process = subprocess.Popen(
                ["mpiexec", "-N", str(num_worker), "python", dotp_code, data_file,
                variable_name, str(N), str(D), 'dot', str(c), "True", "False",
                str(n), str(nN), dotp_file, "True", "True", "True"],
                stdout=subprocess.PIPE,  # Capture standard output
                stderr=subprocess.PIPE,  # Capture errors
                text=True,  # Output as text (instead of bytes)
                bufsize=1,  # Line buffering (ensures real-time output)
            )

            # Read and print live output
            for line in process.stdout:
                print(line.strip())  # Process each line of output in real-time
                if (self.completed_percent < 45):
                    self.completed_percent += 2
                    self.progress_bar.setValue(self.completed_percent)

            process.wait()

            t1 = time.time()
            print("Dot Product Done in {0:.2f} seconds".format(t1 - t0))
            cleanup_code = cxfel_root + "/sna/post_sna_cleanup_.py"
            process = subprocess.Popen(
                ["python", cleanup_code, 'data_chunk', data_file, str(n), str(c)],
                stdout=subprocess.PIPE,  # Capture standard output
                stderr=subprocess.PIPE,  # Capture errors
                text=True,  # Output as text (instead of bytes)
                bufsize=1,  # Line buffering (ensures real-time output)
            )

            # Read and print live output
            for line in process.stdout:
                print(line.strip())  # Process each line of output in real-time
                if (self.completed_percent < 50):
                    self.completed_percent += 1
                    self.progress_bar.setValue(self.completed_percent)

            process.wait()

            if c > 1:

                process = subprocess.Popen(
                    ["python", cleanup_code, 'pipe', 'dummy', str(n), str(c)],
                    stdout=subprocess.PIPE,  # Capture standard output
                    stderr=subprocess.PIPE,  # Capture errors
                    text=True,  # Output as text (instead of bytes)
                    bufsize=1,  # Line buffering (ensures real-time output)
                )

                # Read and print live output
                for line in process.stdout:
                    print(line.strip())  # Process each line of output in real-time
                    if (self.completed_percent < 55):
                        self.completed_percent += 1
                        self.progress_bar.setValue(self.completed_percent)

                process.wait()

            t2 = time.time()
            print("Post SnA Cleanup Done in {0:.2f} seconds".format(t2 - t1))

            ell = self.nEigs
            num_copy = self.num_copy
            nlsa_code = cxfel_root + "/nlsa/run_nlsa.py"

            self.nlsa_file = f"usv_nlsa_N{self.N}_nN{nN_1}_c{self.concat_num}_sigma{int(sigma_1)}_nCopy{self.num_copy}.h5"
            
            mu_psi_file = 'mu_psi.h5'
            process = subprocess.Popen(
                ["python", nlsa_code, data_file, variable_name, mu_psi_file, str(ell), str(N), str(D), str(n), str(c),
                str(num_copy),str(self.nlsa_file)],
                stdout=subprocess.PIPE,  # Capture standard output
                stderr=subprocess.PIPE,  # Capture errors
                text=True,  # Output as text (instead of bytes)
                bufsize=1,  # Line buffering (ensures real-time output)
            )
            for line in process.stdout:
                print(line.strip())  # Process each line of output in real-time
                if (self.completed_percent < 90):
                    self.completed_percent += 1
                    self.progress_bar.setValue(self.completed_percent)

            process.wait()

            process = subprocess.Popen(
                ["python", cleanup_code, 'square', 'dummy', str(n), str(c)],
                stdout=subprocess.PIPE,  # Capture standard output
                stderr=subprocess.PIPE,  # Capture errors
                text=True,  # Output as text (instead of bytes)
                bufsize=1,  # Line buffering (ensures real-time output)
            )

            # Read and print live output
            for line in process.stdout:
                print(line.strip())  # Process each line of output in real-time
                if (self.completed_percent < 95):
                    self.completed_percent += 1
                    self.progress_bar.setValue(self.completed_percent)

            process.wait()

            t7 = time.time()
            print("NLSA Done in {0:.2f} seconds".format(t7 - t2))

            # get the directory of the file
            script_dir = os.path.dirname(os.path.abspath(__file__))
            v_nlsa = np.array(h5py.File(self.nlsa_file)['V'])
            s_nlsa = np.array(h5py.File(self.nlsa_file)['S'])

            for i in range(self.nEigs + 1):
                ax0 = plt.subplot2grid((self.nEigs + 1, 1), (i, 0))
                ax0.plot(v_nlsa[:, i])
                ax0.set_ylabel(r'$V_{}$'.format(i + 1))
                if i == 0:
                    ax0.set_title(r'NLSA Chronos for nN={}, $\sigma_{{factor}}$={}'.format(self.nN_2, self.sigma_2))
                if i < (self.nEigs):
                    ax0.set_xticks([])
                    ax0.tick_params(axis='y', labelsize=14)
                else:
                    ax0.set_xlabel('Time Delay')
                    ax0.tick_params(axis='both', labelsize=14)
            plt.tight_layout()
            plt.savefig('NLSA_chronos.png')
            plt.close()

            plt.scatter(np.arange(self.nEigs + 1), np.diagonal(np.real(s_nlsa)), s=60)
            plt.plot(np.arange(self.nEigs + 1), np.diagonal(np.real(s_nlsa)), linestyle='dashed')
            plt.xlabel('Singular Value Index')
            plt.ylabel('Singular Value')
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)     
            plt.title(r'NLSA Singular Values for nN={}, $\sigma_{{factor}}$={}'.format(self.nN_2, self.sigma_2))
            plt.savefig('NLSA_sing_vals.png')
            plt.close()

            self.tab3 = QWidget()
            tab3_layout = QHBoxLayout(self.tab3)

            self.label1 = FigureCanvas(plt.Figure(figsize=(8, 12), constrained_layout=True))
            self.label1.figure.clear()
            self.label1.figure.add_gridspec(self.nEigs + 1, 1, hspace=0.05)
            self.axes = self.label1.figure.subplots(self.nEigs + 1, 1)

            self.red_boxes = []

            for i in range(self.nEigs + 1):
                ax0 = self.axes[i]
                ax0.plot(v_nlsa[:, i], color='tab:green')
                ax0.set_ylabel(r'$V_{}$'.format(i + 1), fontsize=16)
                if i == 0:
                    ax0.set_title(r'NLSA Chronos for nN={}, $\sigma_{{factor}}$={}'.format(self.nN_2, self.sigma_2))
                    ax0.set_ylim(-1.5, 1.5)
                if i < (self.nEigs):
                    ax0.set_xticks([])
                    ax0.tick_params(axis='y', labelsize=14)
                else:
                    ax0.set_xlabel('Time Delay', fontsize=16)
                    ax0.tick_params(axis='both', labelsize=14)
                    
                    
                self.label1.figure.tight_layout()
                self.label1.figure.subplots_adjust(top=0.96, bottom=0.08, left=0.15, right=0.96)  # Adjust margins
                self.label1.draw()          
                    
                # Get axis limits
                x_min, x_max = ax0.get_xlim()
                y_min, y_max = ax0.get_ylim()

                # Width and height based on full axes span
                width = x_max - x_min
                height = y_max - y_min

                # Create red border rectangle
                red_box = Rectangle(
                    (x_min, y_min), width, height,
                    edgecolor='red',
                    facecolor='none',
                    lw=3,
                    visible=False  # Initially hidden, will be shown on click
                )
                ax0.add_patch(red_box)
                self.red_boxes.append(red_box)

            # Connect click event
            self.label1.mpl_connect("button_press_event", self.on_click)
            self.label1.draw()

            label2 = FigureCanvas(Figure())
            label2.figure.clear()
            ax = label2.figure.add_subplot()
            ax.plot(np.arange(self.nEigs + 1), np.diagonal(np.real(s_nlsa)), linestyle='dashed', color='tab:green')
            ax.scatter(np.arange(self.nEigs + 1), np.diagonal(np.real(s_nlsa)), s=60, color='tab:green')
            ax.set_xlabel('Singular Value Index', fontsize=16)
            ax.set_ylabel('Singular Value', fontsize=16)
            ax.tick_params(axis='both', labelsize=14)
            ### Shift tick labels by +1
            ticks__ = np.arange(len(np.arange(self.nEigs+1)))        
            labels__ = [str(jj + 1) for jj in ticks__]  
            ax.set_xticks(ticks__)
            ax.set_xticklabels(labels__)   
            
            ax.set_title(r'NLSA Singular Values for nN={}, $\sigma_{{factor}}$={}'.format(self.nN_2, self.sigma_2))
            label2.figure.tight_layout()
            label2.draw()
            tab3_layout.addWidget(self.label1)
            tab3_layout.addWidget(label2)

            self.completed_percent = 100
            self.progress_bar.setValue(self.completed_percent)

            self.tabs.addTab(self.tab3, "NLSA of nN="+self.nN_chosen.text()+",\u03C3="+self.sigma_chosen.text())

            self.selected_indices.clear()

    #function to handle click events on chronos graph
    def on_click(self, event):
        """Handles click events to highlight the selected chronos without redrawing everything."""
        
        for i, ax in enumerate(self.axes):
            if ax.contains(event)[0]:
                if i in self.selected_indices:  # Check if the click is inside this axis
                    self.selected_indices.remove(i)
                else:
                    self.selected_indices.add(i)
                self.highlight_selection()
                break
        l=[i+1 for i in self.selected_indices]
        st="".join(str(l))
        self.chronos_selected.setText("chronos selected :"+st)


    def highlight_selection(self):
        for i, rect in enumerate(self.red_boxes):
            if i in self.selected_indices:
                rect.set_visible(True)
            else:
                rect.set_visible(False)
        
        # Redraw the canvas
        if self.label1:
            self.label1.draw_idle()

    def reconstruct(self):
        x_recon = np.array(h5py.File(self.nlsa_file)['X_recon'])

        column_sum = np.zeros_like(x_recon[0])

        for idx in self.selected_indices:
            column_sum += x_recon[idx-1]
        if(len(self.selected_indices)>0):
            #data_reconst_modes_k
            st=("../data_reconst"
                +f"_N{self.N}"
                +f"_nN{int(self.nN_chosen.text())}"
                +f"_c{self.concat_num}"
                +f"_nCopy{self.num_copy}"
                +f"_sigma{int(self.sigma_chosen.text())}"
                +"_modes"
                +"".join(str(sorted([i+1 for i in self.selected_indices])))+".h5"
            )
            # Save to HDF5
            with h5py.File(st, 'w') as f:
                f.create_dataset("selected modes",data=[int(i)+1 for i in self.selected_indices])
                f.create_dataset("data", data=column_sum)
            print("Reconstruction is done and reconstructed data file stored in ",os.path.abspath(st))
            self.chronos_file_msg.setText("NLSA reconstruction is done.")
        else:
            self.chronos_file_msg.setStyleSheet("color: red;")
            self.chronos_file_msg.setText("Select NLSA chronos!")

    
    def click_run(self):

        from misc_tools import write_h5, read_h5
        import os
        import subprocess
        self.h5 = True
        xyz = self.xyz

        sq_code = cxfel_root + "/misc_tools/multi_nN_prepare_dsq.py"
        num_worker = self.mpi
        data_file = 'data_file_for_sna.h5'
        variable_name = self.v_name
        N, D = xyz.shape
        c = self.concat_num

        n = self.n
        nN = self.nn
        sigma_factor = self.sigfac
        sqDist_file = 'dSq'
        cleanup = 'True'
        no_block = 'True'
        run_mpi = 'True'
        if (num_worker < 2): run_mpi = 'False'

        write_h5(data_file, xyz, variable_name)

        process = subprocess.Popen(
            ["mpiexec", "-N", str(num_worker), "python", sq_code, data_file, variable_name, str(N), str(D), 'dSq',
             str(c), 'True', 'False', str(n), str(nN), sqDist_file, cleanup, no_block, run_mpi],
            stdout=subprocess.PIPE,  # Capture standard output
            stderr=subprocess.PIPE,  # Capture errors
            text=True,  # Output as text (instead of bytes)
            bufsize=1,  # Line buffering (ensures real-time output)
        )

        # Read and print live output
        for line in process.stdout:
            print(line.strip())  # Process each line of output in real-time
            if (self.completed_percent < 65):
                self.completed_percent += 1
                self.progress_bar.setValue(self.completed_percent)

        process.wait()
        from ferguson import ferguson_analysis
        sigma_opt_list = []
        for i, nN_i in enumerate(nN):
            sqDist_file_name = sqDist_file + '_nN_' + str(nN_i) + '.h5'
            sigma_opt = ferguson_analysis(sqDist_file_name)
            sigma_opt_list.append(sigma_opt)
            if (self.completed_percent < 75):
                self.completed_percent += int(n / (100 - self.completed_percent * 0.7))
                self.progress_bar.setValue(self.completed_percent)
        write_h5('sigma_opt.h5', sigma_opt_list, 'sigma_opt')

        from diffmap import diffmap_analysis
        nEigs = self.nEigs
        n_nN = len(nN)
        n_sigma = len(sigma_factor)
        eigVec_list = []
        eigVal_list = []
        for i, nN_i in enumerate(nN):
            eigVec_sigma_list = []
            eigVal_sigma_list = []
            sqDist_file_name = sqDist_file + '_nN_' + str(nN_i) + '.h5'
            for j, sigma_j in enumerate(sigma_factor):
                sigma = sigma_opt_list[i] * sigma_j
                eigVec, eigVal, _, _ = diffmap_analysis(sqDist_file_name, sigma, nEigs, 1.0)
                eigVec_sigma_list.append(eigVec)
                eigVal_sigma_list.append(eigVal)
            eigVec_list.append(eigVec_sigma_list)
            eigVal_list.append(eigVal_sigma_list)
            if (self.completed_percent < 85):
                self.completed_percent += int(n_nN * n_sigma / (100 - self.completed_percent) * 0.8)
                self.progress_bar.setValue(self.completed_percent)
        write_h5('eigVec_list.h5', eigVec_list, 'eigVec')
        write_h5('eigVal_list.h5', eigVal_list, 'eigVal')

        eigVec_corr = np.zeros((n_nN * n_sigma, n_nN * n_sigma))
        thresold = 0.90
        for i_nN, _ in enumerate(nN):
            for j_nN, _ in enumerate(nN):
                for i_sigma, _ in enumerate(sigma_factor):
                    ev1 = eigVec_list[i_nN][i_sigma]
                    for j_sigma, _ in enumerate(sigma_factor):
                        ev2 = eigVec_list[j_nN][j_sigma]
                        non_noise = list(set(self.corr_no_noisy(ev1)) & set(self.corr_no_noisy(ev2)))
                        eigVec_corr[i_nN * n_sigma + i_sigma, j_nN * n_sigma + j_sigma] = np.count_nonzero(
                            np.abs(np.dot(ev1.T, ev2)) > thresold)
                        if (self.completed_percent < 99):
                            self.completed_percent += int(n_nN * n_nN * n_sigma * n_sigma / (100 - self.completed_percent) * 0.9)
                            self.progress_bar.setValue(self.completed_percent)

        shuffled = np.swapaxes(eigVec_corr.reshape(n_nN, n_sigma, n_nN, n_sigma), 0, 1)
        shuffled = np.swapaxes(shuffled, 2, 3)
        eigVec_shuffled = shuffled.reshape(n_nN * n_sigma, n_nN * n_sigma)
        del shuffled

        eigVec_heatmap = np.tril(eigVec_shuffled, -1) + np.triu(eigVec_corr, 1)
        write_h5('eigVec_heatmap.h5', eigVec_heatmap, 'eigVec_heatmap')
        write_h5('eigVec_heatmap.h5', eigVec_corr, 'eigVec_corr')
        write_h5('eigVec_heatmap.h5', eigVec_shuffled, 'eigVec_shuffled')

        self.completed_percent = 100
        self.progress_bar.setValue(self.completed_percent)

        return eigVec_heatmap

    def update_heatmap(self, corr_data):
        from matplotlib.ticker import MultipleLocator
        """Update heatmap with calculated data."""
        self.heatmap_canvas.figure.clear()
        self.ax = self.heatmap_canvas.figure.add_subplot()

        alpha_mask = np.ones_like(corr_data)
        np.fill_diagonal(alpha_mask, 0)

        heatmap = self.ax.imshow(corr_data,
                                 extent=[0, len(self.nn) * len(self.sigfac), len(self.nn) * len(self.sigfac), 0],
                                 alpha=alpha_mask) 
        self.ax.set_xlim(0, len(self.nn) * len(self.sigfac))
        self.ax.set_ylim(0, len(self.nn) * len(self.sigfac))
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.xaxis.set_major_locator(MultipleLocator(len(self.sigfac)))
        self.ax.yaxis.set_major_locator(MultipleLocator(len(self.sigfac)))
        self.ax.xaxis.set_minor_locator(MultipleLocator(1))
        self.ax.yaxis.set_minor_locator(MultipleLocator(1))
        self.ax.grid(True, which='both')
        self.ax.set_xlabel('')
        self.ax.set_ylabel('')
        self.ax.set_title('Heatmap', fontsize=14)
        cbar = self.heatmap_canvas.figure.colorbar(heatmap, ticks=np.linspace(0, self.nEigs + 1, 4), ax=self.ax)
        cbar.set_label('Number of Matching Eigenfunctions', fontsize=14)
        cbar.ax.tick_params(labelsize=14)
        self.heatmap_canvas.draw()
        heatmap_name = time.strftime("%Y%m%d_%H%M%S")
        self.heatmap_canvas.figure.savefig(heatmap_name+'heatmap.png')
        self.heatmap_data = corr_data

    def on_heatmap_click(self, event):
        """Handle clicks on the heatmap and display details."""
        if not event.inaxes:
            return

        self.progress_bar.hide()

        x, y = int(event.xdata), int(event.ydata)

        if hasattr(self, 'selected_rect'):
            self.selected_rect.remove()  # Remove previous rectangle

        # Draw a new highlighted rectangle
        self.selected_rect = Rectangle((x, y), 1, 1,linewidth=4, edgecolor='red', facecolor='none')
        # Add to the heatmap axes
        self.ax.add_patch(self.selected_rect)
        self.heatmap_canvas.draw()

        value = int(self.heatmap_data[y, x])
        final_list_u = []
        final_list_l = []
        for i in range(len(self.nn)):
            for j in range(len(self.sigfac)):
                final_list_l.append([self.nn[i], self.sigfac[j]])

        for i in range(len(self.sigfac)):
            for j in range(len(self.nn)):
                final_list_u.append([self.nn[j], self.sigfac[i]])

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setGeometry(500, 600, 300, 200)
        msg.setWindowTitle('Cell Info')
        msg.setStandardButtons(QMessageBox.Close) 

        para_button_1 = None
        para_button_2 = None
        button_1_value = None
        button_2_value = None

        if x > y:
            msg.setText(
                "[nN={}, \u03C3<sub>factor</sub>={}] and [nN={}, \u03C3<sub>factor</sub>={}] have {} matching eigenfunctions.".format(
                    final_list_l[x][0],
                    final_list_l[x][1], final_list_l[y][0], final_list_l[y][1], value))
            button_1_value = [final_list_l[x][0], final_list_l[x][1]]
            button_2_value = [final_list_l[y][0], final_list_l[y][1]]
            para_button_1 = QPushButton("nN={}, \u03C3={}".format(button_1_value[0], button_1_value[1]))

            para_button_2 = QPushButton("nN={}, \u03C3={}".format(button_2_value[0], button_2_value[1]))
        elif x < y:
            msg.setText(
                "[nN={}, \u03C3<sub>factor</sub>={}] and [nN={}, \u03C3<sub>factor</sub>={}] have {} matching eigenfunctions.".format(
                    final_list_u[x][0],
                    final_list_u[x][1], final_list_u[y][0], final_list_u[y][1], value))
            button_1_value = [final_list_u[x][0], final_list_u[x][1]]
            button_2_value = [final_list_u[y][0], final_list_u[y][1]]
            para_button_1 = QPushButton("nN={}, \u03C3={}".format(button_1_value[0], button_1_value[1]))

            para_button_2 = QPushButton("nN={}, \u03C3={}".format(button_2_value[0], button_2_value[1]))
        else:
            msg.setText("Select off-diagonal cells.")


        msg.addButton(para_button_1, QMessageBox.ActionRole)
        msg.addButton(para_button_2, QMessageBox.ActionRole)
        msg.exec_()

        if msg.clickedButton() == para_button_1:
            self.nN_chosen.setText(str(button_1_value[0]))
            self.sigma_chosen.setText(str(button_1_value[1]))
        elif msg.clickedButton() == para_button_2:
            self.nN_chosen.setText(str(button_2_value[0]))
            self.sigma_chosen.setText(str(button_2_value[1]))

if __name__ == "__main__":

    #creating temporary directory
    # Generate a timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Define folder name with timestamp
    folder_name = f"temp_data_{timestamp}"

    # get the directory of the file
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Create the folder in the current directory
    temp_dir = os.path.join(script_dir, folder_name)
    os.makedirs(temp_dir, exist_ok=True)

    # Change the current working directory
    os.chdir(temp_dir)  
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setMinimumSize(1200, 800)
    window.show()
    sys.exit(app.exec_())
