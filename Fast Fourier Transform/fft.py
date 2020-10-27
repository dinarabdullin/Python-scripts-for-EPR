'''
fft.py

Permorms FFT

Requirements: Python3, numpy, scipy, matplotlib
'''

import os
import sys
import wx
import math
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
rcParams['font.size'] = 14
linestyles = ['k-', 'r-', 'b-', 'm-', 'c-']


default_parameters = {
    'complex': True, 
    'background_start': -50,
    'appodization': True,
    'zeropadding': 1,
    'scale_first_point': True,   
    'output': 'real'
}


def get_path(message):
    app = wx.App(None) 
    dialog = wx.FileDialog(None, message, wildcard='*.dat', style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
    if dialog.ShowModal() == wx.ID_OK:
        path = dialog.GetPath()
    else:
        path=""
    return path


def load_data(filename, column_number):
    x = []
    y = []
    file = open(filename, 'r')
    for line in file:
        str = line.split()
        x.append( float(str[0]) )
        y.append( float(str[column_number]) )
    file.close()
    xv = np.array(x)
    yv = np.array(y)
    return [xv, yv]


def read_fft_parameters():
    parameters = {}
    print('\nEnter the FFT parameters:\n')
    var = input("DC offset: calculated for the last N points of the signal, where N = ... (default: %d)\n" % default_parameters['background_start'])
    if (var == ""):
        parameters['background_start'] = default_parameters['background_start']
    else:
        val = [int(i) for i in var.split(' ')]
        if len(val) == 1:
            if val[0] > 0:
                val[0] = -val[0]
            parameters['background_start'] = val[0]
        else:
            raise ValueError('Illelgible value!')
            sys.exit(1)
    var = input("Appodization: True or False (default: %s)\n" % default_parameters['appodization'])
    if (var == ""):
        parameters['appodization'] = default_parameters['appodization']
    else:
        if (var == 'True') or (var == 'true'):
            parameters['appodization'] = True
        elif (var == 'False') or (var == 'false'):
            parameters['appodization'] = False
        else:
            raise ValueError('Illelgible value!')
            sys.exit(1)
    var = input("Zero filling: [entered value] x [number of data points] (default: %d)\n" % default_parameters['zeropadding'])
    if (var == ""):
        parameters['zeropadding'] = default_parameters['zeropadding']
    else:
        val = [int(i) for i in var.split(' ')]
        if len(val) == 1:
            if val[0] >= 0:
                parameters['zeropadding'] = val[0]
            else:
                raise ValueError('Illelgible value!')
                sys.exit(1)
        else:
            raise ValueError('Illelgible value!')
            sys.exit(1)
    var = input("Scale first data point: True or False (default: %s)\n" % default_parameters['scale_first_point'])
    if (var == ""):
        parameters['scale_first_point'] = default_parameters['scale_first_point']
    else:
        if (var == 'True') or (var == 'true'):
            parameters['scale_first_point'] = True
        elif (var == 'False') or (var == 'false'):
            parameters['scale_first_point'] = False
        else:
            raise ValueError('Illelgible value!')
            sys.exit(1)
    var = input("Output data format: original, real, imaginary, absolute (default: %s)\n" % default_parameters['output'])
    if (var == ""):
        parameters['output'] = default_parameters['output']
    else:
        if (var == 'Original') or (var == 'original'):
            parameters['output'] = 'original'
        elif (var == 'Real') or (var == 'real'):
            parameters['output'] = 'real'
        elif (var == 'imaginary') or (var == 'imaginary'):
            parameters['output'] = 'imaginary'
        elif (var == 'Absolute') or (var == 'absolute'):
            parameters['output'] = 'absolute' 
        else:
            raise ValueError('Illelgible value!')
            sys.exit(1)
    return parameters


def dc_offset_correction(sig, bckg_start):
    bckg_level = np.mean(sig[-bckg_start:])
    bckg = bckg_level * np.ones(sig.size)
    sig_shifted = sig - bckg
    return [sig_shifted, bckg]


def appodization(sig, enable):
    sig_app = []
    if (enable):
        hamming = np.hamming(2*sig.size-1)
        window = hamming[-sig.size:]
        sig_app = window * sig
    else:
        window = np.ones(sig.size)
        sig_app = sig
    return [sig_app, window]


def zeropadding(t, sig, length = 0):
    t_zp = []
    sig_zp = []
    if (length == 0):
        t_zp = t
        sig_zp = sig
    else:
        tmax = t[0] + (t[1]-t[0]) * float((length+1)*t.size-1)
        t_zp = np.linspace(t[0], tmax, (length+1)*t.size)
        sig_zp = np.append(sig, np.zeros(length*sig.size))
    return [sig_zp, t_zp]


def scale_first_point(sig, enable):
    sig_scaled = []
    if (enable):
        sig_scaled = sig
        sig_scaled[0] = 0.5 * sig[0]
    else:
        sig_scaled = sig
    return sig_scaled
    

def FFT(t, sig, parameters): 
    # Substract the DC offset
    [sig_shifted, bckg] = dc_offset_correction(sig, parameters['background_start'])
    #plot_signal([t, t, t], [np.real(sig), np.real(bckg), np.real(sig_shifted)], ['signal', 'background', 'signal - background'])
    # Apply the appodization
    sig_app, window = appodization(sig_shifted, parameters['appodization'])
    #plot_signal([t, t, t], [np.real(sig_shifted), window, np.real(sig_app)], ['signal', 'window', 'signal * window']) 
    # Apply the zero-padding
    [sig_zp, t_zp] = zeropadding(t, sig_app, parameters['zeropadding'])
    #plot_signal([t, t_zp], [np.real(sig_app), np.real(sig_zp)], ['signal', 'signal + zeros'])
    # Scale first point
    sig_scaled = scale_first_point(sig_zp, parameters['scale_first_point'])
    # Apply FFT
    spc = np.fft.fft(sig_scaled)
    dt = t[1] - t[0]
    f = np.fft.fftfreq(spc.size, dt)
    # Re-organize the data
    f_sorted = np.fft.fftshift(f)
    spc_sorted = np.fft.fftshift(spc)
    # Calculate Re, Im, and Abs of the spectrum
    spc_re = np.real(spc_sorted)
    spc_im = np.imag(spc_sorted)
    spc_abs = np.abs(spc_sorted)
    # Set the data to be outputed
    if (parameters['output'] == 'original'):
        spc_output = spc
    elif (parameters['output'] == 'real'):
        spc_output = spc_re
    elif (parameters['output'] == 'imaginary'):
        spc_output = spc_im
    elif (parameters['output'] == 'absolute'):
        spc_output = spc_abs
    # Plot the result
    plot_spectrum([f_sorted], [spc_output/np.amax(spc_output)])
    return [f_sorted, spc_output]


def plot_signal(X, Y, legend_text = []):
    xmin = np.amin([np.amin(x) for x in X])
    xmax = np.amax([np.amax(x) for x in X])
    ymin = np.amin([np.amin(y) for y in Y])
    ymax = np.amax([np.amax(y) for y in Y])
    fig = plt.figure(facecolor='w', edgecolor='w')
    axes = fig.gca()
    for i in range(len(X)):
        axes.plot(X[i], Y[i], linestyles[i])
    axes.set_xlim(xmin, xmax)
    axes.set_ylim(ymin-0.1, ymax+0.1)
    axes.set_xlabel(r'$\mathit{t}$ ($\mathit{\mu s}$)')
    axes.set_ylabel(r'$Signal$')
    if not (legend_text == []):
        axes.legend(legend_text)
    plt.tight_layout()
    plt.draw()
    plt.show(block=False)


def plot_spectrum(X, Y, legend_text = []):
    xmin = np.amin([np.amin(x) for x in X])
    xmax = np.amax([np.amax(x) for x in X])
    ymin = np.amin([np.amin(y) for y in Y])
    ymax = np.amax([np.amax(y) for y in Y])
    fig = plt.figure(facecolor='w', edgecolor='w')
    axes = fig.gca()
    for i in range(len(X)):
        axes.plot(X[i], Y[i], linestyles[i])
    axes.set_xlim(xmin, xmax)
    axes.set_ylim(ymin-0.1, ymax+0.1)
    axes.set_xlabel(r'$\mathit{f}$ ($\mathit{MHz}$)')
    axes.set_ylabel(r'$Amplitude$')
    if not (legend_text == []):
        axes.legend(legend_text)
    plt.tight_layout()
    plt.draw()
    plt.show(block=False)
    return [fig, axes]
    

def keep_figures_live():
	plt.show()


def save_data(filename, x, y):
    file = open(filename, 'w')
    if isinstance(y[0], complex):
        for i in range(x.size):    
            file.write('{0:<12.6f}'.format(x[i]))
            file.write('{0:<12.6f}'.format(np.real(y[i])))
            file.write('{0:<12.6f}'.format(np.imag(y[i])))
            file.write('\n')  
    else:
        for i in range(x.size):    
            file.write('{0:<12.6f}'.format(x[i]))
            file.write('{0:<12.6f}'.format(y[i]))
            file.write('\n')  
    file.close()


if __name__ == '__main__':
    # Load the signal
    filename_signal = get_path("Load the signal")
    if (filename_signal == ""):
        raise ValueError('Could not load the signal!')
        sys.exit(1)
    else:
        print('\nThe time trace is loaded from \n%s' % filename_signal)
        [t, sig] = load_data(filename_signal, 1)
    # Enter the FFT parameters
    parameters = read_fft_parameters() 
    # Calculate the FFT of the signal
    [f, spc] = FFT(t, sig, parameters)
    # Save the spectrum
    filename_spectrum = os.path.splitext(filename_signal)[0] + "_fft.dat"
    save_data(filename_spectrum, f, spc)
    keep_figures_live()