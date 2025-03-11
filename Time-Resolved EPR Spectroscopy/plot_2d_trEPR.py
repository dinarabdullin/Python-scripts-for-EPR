#!/usr/bin/env python

'''plot_2d_trEPR.py: Plots and saves the time-resolved EPR data, allows filtering the time axis.'''
__author__ = "Dinar Abdullin"
__copyright__ = "Copyright 2025, University of Bonn"


import os
import sys
import wx
import math
import numpy as np
import scipy.signal
from copy import deepcopy
from numpy import unravel_index
import time
import datetime
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['xtick.direction'] = 'out'
rcParams['ytick.direction'] = 'out'
rcParams['axes.facecolor']= 'white'
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = 'Arial'
rcParams['lines.linewidth'] = 2
rcParams['xtick.major.size'] = 8
rcParams['xtick.major.width'] = 1.5
rcParams['ytick.major.size'] = 8
rcParams['ytick.major.width'] = 1.5
rcParams['font.size'] = 12


''' Data file structure '''
columns = {
    't': 1, 
    'b': 2, 
    'i_re': 3, 
    'i_im': 4
    }
scale_factors = {
    't': 1e-3, # ns to us
    'b': 0.1, # Gauss to mT 
    'i_re': 1.0, 
    'i_im': 1.0
    }
skip_rows = 2
axes_order = ['t', 'b'] # The values of the first axis are incremented first.
column_names = {'t': 'Time [ns]', 'b': 'Field [G]', 'i_re': 'Intensity(real)', 'i_im': 'Intensity(imag)'}


def get_user_input(question, answer_datatype = str, default_answer = '', answer_options = {}):
    '''
    Uses Q&A to get the user input.
    
    Arguments:
    question -- A question addressed to the user.
    answer_datatype -- The expected data type of an answer (atr, int, float).
    default_answer - A default answer.
    answer_options -- A dictionary with possible anwers (optinal).
    
    Returns:
    accepted_answer -- The answer provided by the user.
    '''
    var = input(question)
    if not var:
        if answer_options:
            accepted_answer = answer_options[default_answer]
        else:
            accepted_answer = default_answer
    else:
        if isinstance(answer_datatype, str):
            entered_answer = var
            if answer_options:
                if entered_answer in answer_options:
                    accepted_answer = answer_options[entered_answer]
                else:
                    raise ValueError("Unexpected answer!")
                    sys.exit(1)
            else:
                accepted_answer = entered_answer
        else:
            try:
                entered_answer = answer_datatype(var)
                if answer_options:
                    if entered_answer in answer_options:
                        accepted_answer = answer_options[entered_answer]
                    else:
                        raise ValueError("Unexpected answer!")
                        sys.exit(1)
                else:
                    accepted_answer = entered_answer
            except (ValueError, TypeError):
                raise ValueError("Unexpected answer!")
                sys.exit(1)
    sys.stdout.write("User input: {0:s}\n".format(str(accepted_answer)))
    return accepted_answer


def get_path(message):
    ''' 
    Pop-ups window to get a file path.
    
    Arguments:
    message -- The title of a pop-up window.
    
    Returns:
    path -- The path to a selected file.
    '''
    app = wx.App(None) 
    dialog = wx.FileDialog(None, message, wildcard = '*.*', style = wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
    if dialog.ShowModal() == wx.ID_OK:
        path = dialog.GetPath()
    else:
        path = ""
    return path


def read_data(filepath, columns, skip_rows, scale_factors):
    ''' 
    Load a 2d time-resolved EPR spectrum from a file.
    
    Arguments:
    filepath -- The path to a data file with a time-resolved EPR spectrum.
    columns -- The column numbers to be read.
    skip_rows -- The number of rows to be skipped.
    scale_factors -- Scale factor for each column.
    
    Returns:
    data -- The data of a time-resolved EPR spectrum: 
            't' - time axis
            'b' - magnetic field axis
            'i_re' - signal intensities, in-phase component
            'i_im' - signal intensities, quadrature component
    '''
    data = {}
    with open(filepath, 'r') as f:
        for column_name, column_idx in columns.items():
            v = [float(line.split()[column_idx]) * scale_factors[column_name] for line in f.readlines()[skip_rows:]]
            data[column_name] = v
            f.seek(0)
    return data


def save_data(filepath, data, columns, column_names, scale_factors):
    ''' 
    Load a 2d time-resolved EPR spectrum from a file.
    
    Arguments:
    filepath -- The path to which the data file will be saved.
    data -- The data of a time-resolved EPR spectrum: 
            't' - time axis
            'b' - magnetic field axis
            'i_re' - signal intensities, in-phase component
            'i_im' - signal intensities, quadrature component
    '''
    n = len(data[columns[0]])
    with open(filepath, 'w') as f:
        for column in columns:
            f.write("{0:<20s}".format(column_names[column]))
        f.write("\n")
        for i in range(n):
            for column in columns:
                f.write("{0:<20.0f}".format(data[column][i] / scale_factors[column]))
            f.write("\n")
        f.close()
               

def table2matrix(table, axes_order):
    '''
    Converts a table into a matrix.
    
    Arguments:
    table -- A 2D time-resolved EPR spectrum stored as a table:
             't' - time axis [n,]
             'b' - magnetic field axis [n,]
             'i_re' - signal intensities, in-phase component [n,]
             'i_im' - signal intensities, quadrature component [n,]
    axes_order -- The order in which the values of 't' and 'b' are incremented. The values of the first axis are incremented first.
    
    Returns:
    matrix -- A 2D time-resolved EPR spectrum stored as a matrix:
              't' - time axis [1, n_t]
              'b' - magnetic field axis [1, n_b]
              'i_re' - signal intensities, in-phase component [n_b, n_t]
              'i_im' - signal intensities, quadrature component  [n_b, n_t]
    
    n = n_t * n_b
    '''
    n_b = table['t'].count(table['t'][0])
    n_t = table['b'].count(table['b'][0])
    sys.stdout.write("The spectrum dimensions are {0:d} (field) x {1:d} (time).\n".format(n_b, n_t))
    matrix = {}
    if axes_order[0] == 't':
        v_t = table['t'][:n_t]
        v_b = [table['b'][n_t * i] for i in range(n_b)]
        m_i_re = [table['i_re'][n_t * i : n_t * (i + 1)] for i in range(n_b)]
        m_i_im = [table['i_im'][n_t * i : n_t * (i + 1)] for i in range(n_b)]
        matrix = {
            't': np.array(v_t), 
            'b': np.array(v_b), 
            'i_re': np.array(m_i_re), 
            'i_im': np.array(m_i_im)
            }
    elif axes_order[0] == 'b':
        v_b = table['b'][:n_b]
        v_t = [table['t'][n_b * i] for i in range(n_t)]
        m_i_re = [table['i_re'][n_b * i : n_b * (i + 1)] for i in range(n_t)]
        m_i_im = [table['i_im'][n_b * i : n_b * (i + 1)] for i in range(n_t)]
        matrix = {
            't': np.array(v_t), 
            'b': np.array(v_b), 
            'i_re': np.transpose(np.array(m_i_re)), 
            'i_im': np.transpose(np.array(m_i_im))
            }
    else:
        raise ValueError("Unexpected value of the axis name!")
        sys.exit(1)
    return matrix


def matrix2table(matrix):
    '''
    Converts a table into a matrix.
    
    Arguments:
    matrix -- A 2D time-resolved EPR spectrum stored as a matrix:
              't' - time axis [1, n_t]
              'b' - magnetic field axis [1, n_b]
              'i_re' - signal intensities, in-phase component [n_b, n_t]
              'i_im' - signal intensities, quadrature component  [n_b, n_t]

    Returns:
    table -- A 2D time-resolved EPR spectrum stored as a table:
             't' - time axis [n,]
             'b' - magnetic field axis [n,]
             'i_re' - signal intensities, in-phase component [n,]
             'i_im' - signal intensities, quadrature component [n,]
    
    n = n_t * n_b
    '''
    n_t, n_b = matrix['t'].size, matrix['b'].size
    vt, vb, m_i_re, m_i_im = spc_2d['t'], spc_2d['b'], spc_2d['i_re'], spc_2d['i_im']
    table = {'t': [], 'b': [], 'i_re': [], 'i_im': []}
    for i in range(n_b):
        table['t'] += vt.tolist()
        table['b'] += [vb[i] for j in range(n_t)]
        table['i_re'] += m_i_re[i, :].tolist()
        table['i_im'] += m_i_im[i, :].tolist()
    return table


def get_projections(spc_2d, pos):
    '''
    Gets the projections of a time-resolved EPR spectrum onto the time and field axes.
    
    Arguments:
    spc_2d -- A 2D time-resolved EPR spectrum stored as a matrix.  
    pos -- The position at which the projections are read out.
    
    Returns:
    spc_t -- The 1D projection of a time-resolved EPR spectrum onto the time axis.
             't' - time axis [1, n_t]
             'i_re' - signal intensities, in-phase component [1, n_t]
             'i_im' - signal intensities, quadrature component  [1, n_t]
    spc_b -- The 1D projection of a time-resolved EPR spectrum onto the field axis.
             'b' - field axis [1, n_b]
             'i_re' - signal intensities, in-phase component [1, n_b]
             'i_im' - signal intensities, quadrature component [1, n_b]
    '''
    # Get the position
    vt, vb, m_i_re, m_i_im = spc_2d['t'], spc_2d['b'], spc_2d['i_re'], spc_2d['i_im']
    if pos == "maxabs":
        pos_idx = unravel_index(np.argmax(np.abs(m_i_re)), m_i_re.shape)
    elif pos == "max":
        pos_idx = unravel_index(np.argmax(m_i_re), m_i_re.shape)
    elif pos == "min":
        pos_idx = unravel_index(np.argmin(m_i_re), m_i_re.shape)
    else:
        raise ValueError("Unexpected value of the spectral position!")
        sys.exit(1)
    # Read out the projections
    spc_t = {
        't': vt, 
        'i_re': m_i_re[pos_idx[0], :],
        'i_im': m_i_im[pos_idx[0], :]
    }
    spc_b = {
        'b': vb, 
        'i_re': np.transpose(m_i_re[:, pos_idx[1]]),
        'i_im': np.transpose(m_i_im[:, pos_idx[1]])
    }
    return spc_t, spc_b


def apply_bandpass_filter(spc_2d, param):
    '''
    Applies a band-pass filter to the time-resolved EPR spectrum.
    
    Arguments:
    spc_2d -- A 2D time-resolved EPR spectrum. 
    param -- The parameters of a band-pass filter
             'f0' - centre frequency 
             'bw' - bandwidth determined at -3 dB
             'nharm' - the number of harmonics
             'order' - the order of the Butterworth filter
    
    Returns:
    spc_2d_filtered -- A filtered 2D time-resolved EPR spectrum.
    '''
    f0, bw, nharm, order = param['f0'], param['bw'], param['nharm'], param['order']
    fs = 1 / (spc_2d['t'][1] - spc_2d['t'][0])
    spc_2d_filtered = deepcopy(spc_2d)
    for i in range(nharm + 1):
        Wn = ((f0 - 0.5 * bw) * (i + 1), (f0 + 0.5 * bw) * (i + 1))
        sos = scipy.signal.butter(order, Wn, 'bandstop', fs=fs, output='sos')
        spc_2d_filtered['i_re'] = scipy.signal.sosfiltfilt(sos, spc_2d_filtered['i_re'])
        # for j in range(spc_2d_filtered['b'].size):
            # spc_2d_filtered['i_re'][j, :] = scipy.signal.sosfiltfilt(sos, spc_2d_filtered['i_re'][j, :])
            # spc_2d_filtered['i_im'][j, :] = scipy.signal.sosfiltfilt(sos, spc_2d_filtered['i_im'][j, :])
    return spc_2d_filtered
    

def plot_trepr_data(spc_2d, spc_t, spc_b, filepath = None):
    '''
    Plots a 2d time-resolved EPR spectrum in 1- and 2-dimentions.
    
    Arguments:
    spc_2d -- A time-resolved EPR spectrum stored as a matrix.  
    spc_t -- The projection of a time-resolved EPR spectrum onto the time axis.
    spc_b -- The projection of a time-resolved EPR spectrum onto the field axis.
    '''
    fig = plt.figure(figsize=(10,10), facecolor='w', edgecolor='w')
    # 2d time-resolved EPR spectrum with x = time and y = field
    ax1 = fig.add_subplot(2, 2, 1)
    im1, _ = plot_spectrum_2d(spc_2d, ['t', 'b'], ax1)
    # 2d time-resolved EPR spectrum with x = field and y = time
    ax2 = fig.add_subplot(2, 2, 2)
    im2, _ = plot_spectrum_2d(spc_2d, ['b', 't'], ax2)
    # 1d time-resolved EPR spectrum with x = time at the maximal intensity
    ax3 = fig.add_subplot(2, 2, 3)
    im3, _ = plot_spectrum_1d(spc_t, 't', ax3)
    # 1d time-resolved EPR spectrum with x = field at the maximal intensity
    ax4 = fig.add_subplot(2, 2, 4)
    im4, _ = plot_spectrum_1d(spc_b, 'b', ax4)
    plt.tight_layout()
    if filepath is not None:
        fig.savefig(filepath, format='png', dpi=600)
    plt.show()


def plot_spectrum_2d(spc_2d, axes_labels, figure_axes=None):
    '''
    Plots a time-resolved EPR spectrum in 2D.
    
    Arguments:
    spc_2d -- A time-resolved EPR spectrum stored as a matrix.
    axes_labels -- The axes of the spectrum. The first axis is assigned to x, the second to y.
    figure_axes -- The figure axes used for plotting.
    '''
    # Set the figure axes
    if figure_axes is None:
        fig = plt.figure(figsize=(10,12), facecolor='w', edgecolor='w')
        ax = fig.gca()
    else:
        ax = figure_axes
    # Prepare data for plotting
    dt = spc_2d['t'][1] - spc_2d['t'][0]
    vt = spc_2d['t'] - 0.5 * dt
    vt = np.append(vt, vt[-1] + dt)
    db = spc_2d['b'][1] - spc_2d['b'][0]
    vb = spc_2d['b'] - 0.5 * db
    vb = np.append(vb, vb[-1] + db)
    m_i_re = spc_2d['i_re']
    # Set grids & plot settings
    if axes_labels[0] == 't':
        x, y = np.meshgrid(vt, vb)
        z = m_i_re / np.max(np.abs(m_i_re))
        x_label = r'Time ($\mu s$)'
        y_label = 'Magnetic field (mT)'
        x_lim = [spc_2d['t'][0], spc_2d['t'][-1]]
        y_lim = [spc_2d['b'][0], spc_2d['b'][-1]]
    elif axes_labels[0] == 'b':
        x, y = np.meshgrid(vb, vt)
        z = np.transpose(m_i_re) / np.max(np.abs(m_i_re))
        x_label = 'Magnetic field (mT)'
        y_label = r'Time ($\mu s$)'
        x_lim = [spc_2d['b'][0], spc_2d['b'][-1]]
        y_lim = [spc_2d['t'][0], spc_2d['t'][-1]]
    else:
        raise ValueError("Unexpected value of the axis name!")
        sys.exit(1)    
    # Plot
    im = ax.pcolormesh(x, y, z, vmin=-1, vmax=1, cmap = 'RdBu_r')
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.colorbar(im, ax = ax, ticks=[-1, 0, 1])
    return im, ax


def plot_spectrum_1d(spc_1d, axis_label, figure_axes=None):
    '''
    Plots a time-resolved EPR spectrum in 2D.
    
    Arguments:
    spc_2d -- A time-resolved EPR spectrum stored as a matrix.
    axes_labels -- The axis of the spectrum.
    figure_axes -- The figure axes used for plotting.
    '''
    # Set the figure axes
    if figure_axes is None:
        fig = plt.figure(figsize=(10,10), facecolor='w', edgecolor='w')
        ax = fig.gca()
    else:
        ax = figure_axes
    # Prepare data for plotting
    if axis_label == 't':
        x = spc_1d['t']
        y = spc_1d['i_re'] / np.amax(spc_1d['i_re'])
        x_label = r'Time ($\mu s$)'
    elif axis_label == 'b':
        x = spc_1d['b']
        y = spc_1d['i_re'] / np.amax(spc_1d['i_re'])
        x_label = 'Magnetic field (mT)'
    # Plot
    im = ax.plot(x, y, color = "black")
    ax.set_xlim([x[0], x[-1]])
    ax.set_xlabel(x_label)
    ax.set_ylabel('Norm. intensity')
    return im, ax


if __name__ == "__main__":
    # Load a time-resolved EPR spectrum
    filepath_spc = get_path("Load a TR-EPR spectrum")
    if filepath_spc == "":
        raise ValueError("No file could be loaded!")
        sys.exit(1)
    else:
        filedir_spc, filename_spc = os.path.split(filepath_spc)
        filename_spc = os.path.splitext(filename_spc)[0]
        data = {}
        data = read_data(filepath_spc, columns, skip_rows, scale_factors)
        spc_2d = table2matrix(data, axes_order)
    # Get the spectrum projections onto the time and field axes
    pos = get_user_input(
        "\nRead out the 1D TR-EPR spectrum at the maximum of the absolute intensity (type 0), or at the intensity maximum (type 1), or at the intensity minimum (type 2) [default value: 0]: ",
        int, 
        0,
        {0: 'maxabs', 1: 'max', 2: 'min'}
        )
    spc_t, spc_b = get_projections(spc_2d, pos)
    # Save a time-resolved EPR spectrim in 1D and 2D
    filepath_data = filedir_spc + '/' + filename_spc + '.dat'
    save_data(filepath_data, data, ['t', 'b', 'i_re', 'i_im'], column_names, scale_factors)
    filepath_data_1d = filedir_spc + '/' + filename_spc + '_1d' + '.dat'
    save_data(filepath_data_1d, spc_b, ['b', 'i_re', 'i_im'], column_names, scale_factors)
    # Plot a time-resolved EPR spectrum
    filepath_img = filedir_spc + '/' + filename_spc + '_orig' + '.png'
    plot_trepr_data(spc_2d, spc_t, spc_b, filepath_img)
    # Apply a band-pass filter
    bpf_flag = get_user_input(
        "\nApply a band-pass filter to remove background oscillation - type y or n [default value: y]: ",
        str,
        'y',
        {'y': 'yes', 'n': 'no'}
        )
    if bpf_flag:
        bpf_f0 = get_user_input(
            "\nEnter the centre frequency of the filter in MHz [default value: 1.45]: ",
            float, 
            1.45
            )
        bpf_bw = get_user_input(
            "\nEnter the bandwidth (-3 dB) of the filter in MHz [default value: 0.2]:",
            float,
            0.2
            )
        bpf_nharm = get_user_input(
            "\nEnter the number of harmonics to filter [default value: 2]:",
            int,
            2
            )
        bpf_order = get_user_input(
            "\nEnter the order of the Butterworth filter [default value: 5]:",
            int,
            5
            )
        spc_2d_filtered = apply_bandpass_filter(spc_2d, param={'f0': bpf_f0, 'bw': bpf_bw, 'nharm': bpf_nharm, 'order': bpf_order})
        spc_t_filtered, spc_b_filtered = get_projections(spc_2d_filtered, pos)
        data_filtered = matrix2table(spc_2d_filtered)
    # Save a time-resolved EPR spectrim in 1D and 2D
    filepath_data_filtered = filedir_spc + '/' + filename_spc + '_filtered' + '.dat'
    save_data(filepath_data_filtered, data_filtered, ['t', 'b', 'i_re', 'i_im'], column_names, scale_factors)
    filepath_data_1d_filtered = filedir_spc + '/' + filename_spc + '_1d_filtered' + '.dat'
    save_data(filepath_data_1d_filtered, data_filtered, ['b', 'i_re', 'i_im'], column_names, scale_factors)
    # Plot a filtered, time-resolved EPR spectrum
    filepath_img2 = filedir_spc + '/' + filename_spc + '_filtered' + '.png'
    plot_trepr_data(spc_2d_filtered, spc_t_filtered, spc_b_filtered, filepath_img2)
    print('Done!')