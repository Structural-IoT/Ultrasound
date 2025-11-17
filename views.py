from django.shortcuts import render

# Create your views here.
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from numpy import sqrt, sin, cos, pi
from django.shortcuts import render
from django.conf import settings
from .forms import AttenuationForm
import os

# -------------------------------------------------------------
#  Structural IoT LLC
#  www.structuraliot.com
#  Author: Gaofeng
#  Description: Attenuation prediction of ultrasonic wave in polycrystalline materials 
# -------------------------------------------------------------

def plot_attenuation(request):
    image_url = None

    if request.method == 'POST':
        form = AttenuationForm(request.POST)
        if form.is_valid():
            material = form.cleaned_data['material']
            grain_size = form.cleaned_data['grain_size']
            start_freq = form.cleaned_data['start_freq']
            end_freq = form.cleaned_data['end_freq']

            # Material properties
            if material == 'Aluminum':
                c11, c12, c44, des = 103.0, 57.0, 29.0, 2.70
            elif material == 'Steel':
                c11, c12, c44, des = 219.2, 136.8, 109.2, 7.86
            elif material == 'Copper':
                c11, c12, c44, des = 170.0, 120.0, 75.0, 8.96

            c = c11 - c12 - 2 * c44
            ce11 = 3 * c11 / 5 + 6 * c12 / 15 + 12 * c44 / 15
            ce44 = 3 * c11 / 15 - 3 * c12 / 15 + 3 * c44 / 5
            vL = sqrt(ce11 / des)
            vT = sqrt(ce44 / des)

            x = np.linspace(start_freq, end_freq, 100)
            y = np.zeros(100, dtype=float)

            for i in range(len(x)):
                kL = 2 * pi * x[i] / vL
                kT = 2 * pi * x[i] / vT
                xL = kL * grain_size
                xT = kT * grain_size
                aLL_c = xL**4 * c**2 / 2 / des**2 / vL**4 / grain_size
                aLL_i = integrate.quad(
                    lambda thata: (3.0/175 + 2.0/175*cos(thata)**2 + 1.0/525*cos(thata)**4) * sin(thata) / (1.0 + 2 * xL**2 * (1 - cos(thata)))**2, 0, pi
                )
                aLL = aLL_c * float(aLL_i[0])
                aLT_c = xT**4 * c**2 / 2 / des**2 / vL**3 / vT / grain_size
                aLT_i = integrate.quad(
                    lambda thata: (1.0/35 + 2.0/175*cos(thata)**2 - 1.0/525*cos(thata)**4) * sin(thata) / (1.0 + xL**2 + xT**2 - 2 * xL * xT * cos(thata))**2, 0, pi
                )
                aLT = aLT_c * float(aLT_i[0])
                y[i] = (aLL + aLT) * 8.6859  # Convert from Np/mm to dB/mm

            # Save plot to a file
            plt.figure()
            plt.plot(x, y)
            plt.xlabel('Frequency (MHz)')
            plt.ylabel('Attenuation (dB/mm)')
            plt.title(f'Ultrasonic Attenuation vs Frequency for {material}')
            image_path = os.path.join(settings.MEDIA_ROOT, 'attenuation_plot.png')
            plt.savefig(image_path)
            plt.close()

            image_url = os.path.join(settings.MEDIA_URL, 'attenuation_plot.png')

    else:
        form = AttenuationForm()

    return render(request, 'attenuation/form.html', {'form': form, 'image_url': image_url})

