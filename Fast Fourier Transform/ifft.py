'''
ifft.py

Permorms inverse FFT

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
    'zerofilling': 1,
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


def read_ifft_parameters():
    parameters = {}
    return parameters


def crop_signal(x, y):
    x_cropped = []
    y_cropped = []
    Nx = x.size
    Nc = int(float(Nx)/2) + 1
    x_cropped = x[:Nc]
    y_cropped = y[:Nc]
    return [x_cropped, y_cropped]
    

def inverse_FFT(f, spc, parameters): 
    # Format the input data
    f_formated = np.fft.ifftshift(f)
    spc_formated = np.fft.ifftshift(spc)
    #plot_spectrum([f, f_formated], [np.real(spc), np.real(spc_formated)], ['original', 'formated'])
    # Apply inverse FFT
    sig = np.fft.ifft(spc_formated)
    fs = 2 * np.amax(abs(f))
    dt = 1 / fs
    Nt = f.size
    t = np.linspace(0, dt*(Nt-1), Nt)
    # Scale the amplitude
    sig = 2 * sig
    # Normalize the first point to 1
    sig = sig + (1.0-sig[0])*np.ones(sig.size)
    # Crop signal
    [t_cropped, sig_cropped] = crop_signal(t, sig)
    # Calculate Re and Im of the signal
    sig_re = np.real(sig_cropped)
    sig_im = np.real(sig_cropped)
    # Plot the result
    plot_signal([t_cropped, t_cropped], [np.real(sig_cropped), np.imag(sig_cropped)], ['real', 'imaginary'])
    return [t_cropped, sig_cropped]


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
    # Load the spectrum
    filename_spectrum = get_path("Load the spectrum")
    if (filename_spectrum == ""):
        raise ValueError('Could not load the spectrum!')
        sys.exit(1)
    else:
        print('\nThe spectrum is loaded from \n%s' % filename_spectrum)
        [f, spc] = load_data(filename_spectrum, 1) 
    # Enter the iFFT parameters
    parameters = read_ifft_parameters() 
    # Calculate the inverse FFT of the spectrum
    [t, sig] = inverse_FFT(f, spc, parameters)
    # Save the signal
    filename_signal = os.path.splitext(filename_spectrum)[0] + "_ifft.dat"
    save_data(filename_signal, t, sig)
    keep_figures_live()