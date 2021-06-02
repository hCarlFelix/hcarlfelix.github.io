#!/usr/bin/env python
# coding: utf-8

# # PSD analysis of different cases of (simulated) reaction systems
# 
# Some introductionary text...
# 
# 
# ## The basics, linear reactions
# 
# Let's start with the obligatory import of modules we need.

# In[1]:


# Math
import numpy as np
import scipy
from scipy.integrate import odeint
from scipy import signal

# Plotting stuff
import plotly
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots

pio.renderers.default='iframe'


# We'll also need the previously defined `BasicReactionModel` class and `constr_stoichio` function who will construct our system of differential equations (`BasicReactionModel`) and construct a stoichiometry matrix (`constr_stoichio`). These ones have been stored in a module `ModelStuff`.
# 
# I've also defined a template (`PlotStuff`) for how I want to the data to be plotted, resulting in a macro named `plotty` for plotting the data.

# In[2]:


from ModelStuff import *
from PlotStuff import *


# ### Only one desorption step
# 
# To begin, let's look at the following reaction system:
# 
# $$
# \begin{align}
#     A_2 &\rightarrow 2A^*\\
#     B + * &\rightarrow B*\\
#     A^* + B^* &\rightarrow AB^* + *\\
#     AB^* &\rightarrow AB + *
# \end{align}
# $$
# 
# A simple linear reaction scheme, with three adsorption/desorption steps and one surface reaction between two different adsorbates. We'll define the differential equation system by first defining a stoichiometry matrix for the reaction system by the use of `constr_stoichio`. Given this stoichiometry matrix, we can later on define the ODE system by the use of `BasicReactionModel`.
# 
# For this first example, we'll set all reaction constants to be equal (1) except the re-adsorption of AB which we will prohibit by setting its reaction constant (`k[1, -1]`) to 0. Furthermore, we'll set the partial pressures as constant (`constant_term = True` for variables corresponding to pressures) and the initial condition as a clean surface and that we only have one kind of adsorption site.

# In[3]:


reactions = ["A2 + 2* -> 2A*",
             "B + * -> B*",
             "A* + B* -> AB* + *",
             "AB* -> AB + *",
             ]

variables = ["*", "A*", "B*", "AB*", "A2", "B", "AB"]

k = np.array([[1] * len(reactions)] * 2)
k[1, -1] = 0

x0_sites = [1]
x0_surfS = [0] * 3
x0_press = [1] * 3

x0 = x0_sites + x0_surfS + x0_press

constant_term = [False] * (len(x0_sites) + len(x0_surfS)) +                 [True] * len(x0_press)

constant_term[-1] = False

stoichi = constr_stoichio(variables, reactions)

test_model = BasicReactionModel(k.T,
                                stoichi=stoichi,
                                const_term=constant_term,
                                )


# So, we are now ready to solve the differential equation system. For this, we'll use SciPy's solve_ivp function (using the default Runge-Kutta 45 method).

# In[4]:


# Time span
t0 = 0
tf = 10

# Error tolerance
res = 1.0e-12
AbsTol = [res] * test_model.stoichi.shape[1]
RelTol = res

ans = scipy.integrate.solve_ivp(test_model.reaction, 
                                [t0, tf], 
                                x0,
                                rtol=RelTol, 
                                atol=AbsTol,
                                )

# Net reaction rate for each step
r = np.array([test_model.reaction(0, step) for step in ans.y.T]).T

# Plot the data
fig = plotty(ans.t, ans.y, variables[:(len(x0) - len(x0_press))], r[-1])

# Save the plot as an interactive html file for later use
pio.write_html(fig, 'fig1.html')


# Just to make sure that everything behaves correctly, let's try multiplying the reaction constants by 10. As expected, it neither affected the steady-state coverage nor the TOF of the reaction (as the forward and backward reaction cancels each other out) but we reached steady-state much faster.

# In[5]:


k = [[10, 10, 10, 10],
     [10, 10, 10, 0],
     ]
k = np.array(k)

test_model.react_const(k)

t0 = 0
tf = 10

ans = scipy.integrate.solve_ivp(test_model.reaction, 
                                [t0, tf], 
                                x0,
                                rtol=RelTol, 
                                atol=AbsTol,
                                )

r = np.array([test_model.reaction(0, step) for step in ans.y.T]).T

# Plot the data
fig = plotty(ans.t, ans.y, variables[:(len(x0) - len(x0_press))], r[-1])

# Save the plot as an interactive html file for later use
pio.write_html(fig, 'fig2.html')


# ## Adding variation - To modulate variables
# 
# So, having laid the groundwork with by setting up the differential equations for the reaction system, let's move on to the modulation function. Below we have the function `stim_sin` which will allow us to modulate one or more input variables to the ODE solver by the use of a sinusoidal function. Furthermore, it will overwrite the value of the modulated variable while maintaining it as a constant for each loop of the ODE function, ergo we can still prohibit the ODE solver from changing it's value while still adding a modulation function.

# In[6]:


def stim_sin(t, x, func, c):
    '''
    Varies one or more input arguments to a function func with a sinus function.
    
    Its intended use is to add a stimulus function to a function to be solved by
    ODE solver.
    
    func : function
        function describing the ODE system. func(t, x)
    c : list
        The coefficient and constant term of the sinus function.
        x[0] = c[0, 0] * np.sin(c[0, 1] * t + c[0, 2]) + c[0, 3]
        If c[i, 0] = 0, then x[i] = x[i]
    '''
    
    for i, term in enumerate(c):
        if term[0] != 0:
            x[i] = term[0] * np.sin(term[1] * t + term[2]) + term[3]
    
    return func(t, x)


# Except asking for the `t` and `x` variables (used by the ODE solver), `stim_sin` also asks for the ODE function which will be modulated (in this case it is our `test_model.reaction`) and a couple of constants for the sinus function.
# 
# <br>
# 
# $$
# \begin{align}
#     f(x) &= A \, \sin \left(\frac{2 \pi}{\tau} \, t + \varphi \right) + C\\
#     f(x) &= x_1 \, \sin \left(x_2 \, t + x_3 \right) + x_4
# \end{align}
# $$
# 
# <br>
# 
# Let's perform a sine modulation on the H$_2$ gas. As `stim_sin` function will override the initial value for modulated variable, we'll need to define the constants of the sinus function such that our maximum partial pressure will be that of  $x_0$(H$_2$). So, as the sine can only give \[-1, 1\], we simply need to fulfill the conditions of $x_1+x_4=x_0$(H$_2$) and $-x_1+x_4>0$. In this example, we'll let the H$_2$ partial pressure oscillate between 0 and 1, and as such we'll set $x_1=0.5$ and $x_4=0.5$.
# 
# >Note: While ignoring the initial value, `stim_sin` will not affect the value at $t_0$ as it is assigned by the ODE solver. As such, there might be cases in which a solution can not be obtained due to a potential dramatic shift, it is therefore advisable to set the starting condition to equal the sine function at $t_0$.

# In[7]:


period = 0.5#0.637 # Equal to TOF at steady-state
stim_const = np.array([[0] * 4] * len(x0), dtype=float)
stim_const[4, 0] = 0.5
stim_const[4, 1] = 2 * np.pi / period
stim_const[4, 2] = np.pi / 2 # Phase delay correction, start at sin(...) = 1
stim_const[4, 3] = 0.5

tf = 10 * period

ans_sin = scipy.integrate.solve_ivp(stim_sin,
                                    [t0, tf], 
                                    x0,
                                    args=[test_model.reaction, 
                                          stim_const,
                                          ],
                                    rtol=RelTol, 
                                    atol=AbsTol,
                                    )


r = np.array([test_model.reaction(0, step) for step in ans_sin.y.T]).T

fig = plotty(ans_sin.t, ans_sin.y, variables[:(len(x0) - len(x0_press))], r[-1])


# Another modulation that could be performed (experimentally much more easily) is the square wave modulation. One benefit about this is the possibility of studying several frequencies from one single experiment, the importance of this will be discussed in the PSD section later on. For now, let's define a function for this square wave modulation using the `square` function from `scipy.signal`.

# In[8]:


from scipy import signal

def stim_square(t, x, func, c):
    '''
    Varies one or more input arguments to a function func with a square wave 
    function.
    
    Its intended use is to add a stimulus function to a function to be solved by
    ODE solver.
    
    func : function
        function describing the ODE system. func(t, x)
    const : list
        The coefficient and constant term of the sinus function.
        x[0] = c[0, 0] * signal.square(c[0, 1] * t + c[0, 2]) + c[0, 3]
        If const[i, 0] = 0, then x_stim[i] = x[i]
    '''
    
    for i, term in enumerate(const):
        if term[0] != 0:
            x[i] = term[0] * signal.square(term[1] * t + term[2]) + term[3]
    
    return func(t, x)


# The behaviour of this square wave is a bit different than the sine function so we'll also change the modulation parameters and then finally perform the calculation and plot.

# In[9]:


stim_const[4, 0] = 1
stim_const[4, 1] = np.pi / period
stim_const[4, 2] = 0
stim_const[4, 3] = 0

ans_square = scipy.integrate.solve_ivp(stim_sin,
                                       [t0, tf], 
                                       x0,
                                       args=[test_model.reaction, 
                                             stim_const,
                                             ],
                                       rtol=RelTol, 
                                       atol=AbsTol,
                                       )


r = np.array([test_model.reaction(0, step) for step in ans_square.y.T]).T

fig = plotty(ans_square.t, ans_square.y, variables[:(len(x0) - len(x0_press))], r[-1])


# We could then play around with different rate constants and modulation periods and see how the system responds. Keep in mind though that we are ignoring mass transfer limitations, as such the obtained responses by this simulation at high frequencies can differ dramatically from reality due to mass transfer becoming significant.

# ## Testing PSD
# 
# We have now reached the state in which we can produce a simulated reaction system. Let us now try out phase sensitive detection (PSD) on one of these systems. For this example, let's limit the analysis to the usage of the PSD equation (given below) and not involve any in depth analysis using Fourier transform or such.
# 
# $$
# A(\phi^\text{PSD}) = \frac{2}{T} \int^T_0 A(t) \sin(\omega t + \phi^\text{PSD}) \text{d}t
# $$

# In[10]:


from PSD import psd_demodulation as demodulation


# In[11]:


start_t = np.argmin(abs(ans_sin.t - period * 3))

demod_sine = demodulation(ans_sin.y[:-len(x0_press), start_t:].T, 
                          ans_sin.t[start_t:], 
                          period, 
                          kArr=np.array([1]))

fig = plot_psd([i for i in range(360)], 
               demod_sine, 
               variables[:(len(x0) - len(x0_press))], 
               demod_sine,
               )


# In[14]:


start_t = np.argmin(abs(ans_sin.t - period * 3))

demod_sine = demodulation(ans_sin.y[:-len(x0_press), start_t:].T, 
                          ans_sin.t[start_t:], 
                          period * 4, 
                          kArr=np.array([1]))

fig = plot_psd([i for i in range(360)], 
               demod_sine, 
               variables[:(len(x0) - len(x0_press))], 
               demod_sine,
               )


# In[ ]:


def psd_demodulation(A, t, period, kArr=np.array([1]), 
                    phi=np.arange(0, 360)/180.*np.pi, modtype='sin', angleCorr=0):
  
    '''Phase sensitive detection demodulation procedure by the use of a lock
    in amplifier.

    psd_demodulation(A, t, period, kArr, phi, shape, angleCorr)

    Parameters
    ----------
    A : NxM numpy array
      The data to demodulate
    t : 1xM numpy array
      The time data points of the data
    period : float
      The time period of the stimulation
    kArr : 1-dim. numpy array
      Array with demodulation indices. This corresponds to 'k = n' for 
      sinusoidal and 'k = 2n - 1' for square wave. Default value is [1]
    phi : 1-dim numpy array
      The control angles (in radian) for which the PSD should be solved.
      Default is phi = numpy.arange(0, 360)/180.*np.pi, ergo 360 deg, 1 deg step length
    modtype : string
      The type of modulation, sinusoidal ('sin') or square-wave ('sw')
    angleCorr : float
      A starting angle, used to correct for data not starting at 0 rad.
      Default value is 0

    Returns
    -------
    demodulated : numpy array
    An array in the form of IxJxK for which the dimensions corresponds to:
      I: Demodulation angle (default, 360 deg, 1 deg step)
      J: Y-dimension of A
      K: Frequency, k

    References:
    Urakawa, A.; Burgi, T.; Baiker, A. Chem. Eng. Sci. 2008, 63, 4902-4909.
    '''
        
    if A.ndim > 1:
        demodulated = np.zeros((phi.shape[0], A.shape[1], kArr.shape[0]))
    else:
        demodulated = np.zeros((phi.shape[0], kArr.shape[0]))    
    
    # To get average amplitude of a period
    n_periods = (t[-1] - t[0]) / period
    
    for k in range(0, kArr.shape[0]): # k-order
        for angle in range(0, phi.shape[0]): # controller angle

            if modtype.lower() == 'sin':
                s = np.sin(kArr[k] * 2 * np.pi / period * t + phi[angle] + angleCorr)
            elif modtype.lower() == 'sw':
                # As given by Urakawa et al. Chem. Eng. Sci. 63 (2008) 4906
                #
                # Asin and phisin are the amp. and phase delay for a sinusoidal stim. 
                # at (2n-1)omega freq. and Asw and phisw are the amp. and phase delay 
                # for a square-wave (SW) stim. at omega, but obtained at (2n-1)omega.
                # We can then set the following
                #   pi/4*(2n - 1)*Asw = Asin
                #   phisw = phisin
                # The phase delay and amp. for sinusoidal and SW are equivalent 
                # (with a constant for amp.)

                s = 4 / (np.pi * kArr[k]) * np.sin(kArr[k] * 2 * np.pi / period * t 
                                             + phi[angle] 
                                             + angleCorr)
            else:
                raise ValueError("Incorrect waveform, either sinusoidal (''sin'') or square wave (''SW'')")
            
            if A.ndim > 1:
                for wn in range(0, A.shape[1]):
                    demodulated[angle, wn, k] = 2 / period * (np.trapz(A[:, wn] * s)) / n_periods
            else:
                demodulated[angle, k] = 2 / period * (np.trapz(A * s)) / n_periods

    return demodulated


# In[ ]:


ans_sin.t[start_t]


# In[ ]:


ans_sin.t[-1]


# In[ ]:


(ans_sin.t[-1] - ans_sin.t[start_t]) / period


# In[ ]:


time = np.linspace(0, period)
np.trapz(np.sin(time * 2 * np.pi / (1 * period) + np.pi / 2) * np.sin(time * 2 * np.pi / period))


# In[ ]:


start_t = np.argmin(abs(ans_square.t - period * 3))

demod_square = demodulation(ans_square.y[:-len(x0_press), start_t:].T, 
                            ans_square.t[start_t:], 
                            period, 
                            kArr=np.array([i for i in range(1, 10)]),
                            modtype='SW')

fig = plot_psd([i for i in range(360)], 
               demod_square, 
               variables[:(len(x0) - len(x0_press))], 
               demod_square,
               )


# In[ ]:




