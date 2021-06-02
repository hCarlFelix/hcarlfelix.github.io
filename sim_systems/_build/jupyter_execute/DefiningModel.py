#!/usr/bin/env python
# coding: utf-8

# # Basic generator for a function describing a MKM reaction system
# 
# >__Modules used in this example:__ NumPy, SciPy, Matplotlib
# 
# >__Limitations:__ Mean-field approximation, elementary reaction steps, no mass transfer limitation
# 
# ## The reaction system
# 
# See the following reaction system for the hydrogenation of CO$_2$ into CO and CH$_3$OH.
# 
# <br>
# 
# $$
# \begin{align}
#     \text{H$_2$ (g)} + 2∗ &\rightleftharpoons 2\text{H}^∗\tag{0}\\
#     \text{CO}_2 + \text{H}^* &\rightleftharpoons \text{HCOO}^∗\tag{1}\\
#     \text{HCOO}^∗ + \text{H}^* &\rightleftharpoons \text{HCOOH}^∗ + ∗\tag{2}\\
#     \text{HCOOH}^∗ + \text{H}^* &\rightleftharpoons \text{CH$_2$OOH} + ∗\tag{3}\\
#     \text{CH$_2$OOH}^∗ + * &\rightleftharpoons \text{CH$_2$O}^* + \text{OH}^∗\tag{4}\\
#     \text{OH}^∗ + \text{H}* &\rightleftharpoons \text{H$_2$O}^* + ∗\tag{5}\\
#     \text{CH$_2$O}^∗ + \text{H}* &\rightleftharpoons \text{CH$_3$O}^* + ∗\tag{6}\\
#     \text{CH$_3$O}^∗ + \text{H}* &\rightleftharpoons \text{CH$_3$OH}^* + ∗\tag{7}\\
#     \text{CO$_2$ (g)} + * &\rightleftharpoons \text{CO$_2$}^* + ∗\tag{8}\\
#     \text{CO$_2$}^∗ + \text{H}* &\rightleftharpoons \text{COOH}^* + ∗\tag{9}\\
#     \text{COOH}^∗ &\rightleftharpoons \text{CO (g)} + \text{OH}^∗\tag{10}\\
#     \text{CHOO}^∗ + \text{H}^* &\rightleftharpoons \text{CH$_2$OO}^* + ∗\tag{11}\\
#     \text{CH$_2$OO}^∗ + \text{H}^* &\rightleftharpoons \text{CH$_2$OOH}^* + ∗\tag{12}\\
#     \text{H$_2$O}^∗ &\rightleftharpoons \text{H$_2$O (g)} + ∗\tag{13}\\
#     \text{CH$_3$OH}^∗ &\rightleftharpoons \text{CH$_3$OH (g)} + ∗\tag{14}\\
# \end{align}
# $$
# 
# <br>
# 
# We can express the reaction rate for each of these equilibrium reactions by simple elementary steps, such as:
# 
# <br>
# 
# $$
# \begin{align}
#     r_0 &= k_{0,0} \, \theta_\text{∗}^2  \, p_\text{H$_2$}      - k_{1,0} \, \theta_\text{H}^2\\
#     r_1 &= k_{0,1} \, \theta_\text{∗}    \, p_\text{CO$_2$}     - k_{1,1} \, \theta_\text{CHOO}\\
#     \vdots & \\
#     r_{14} &= k_{0,14} \, \theta_\text{CH$_3$OH}   \,           - k_{1,14} \, p_\text{CH$_3$OH}\\
# \end{align}
# $$
# 
# <br>
# 
# Of which $k_{0, i}$ is the reaction constant for the forwards reaction of equilibrium step $i$ and $k_{1, i}$ is that of the backwards reaction.
# 
# <br>
# 
# $$
# \begin{equation}
# k = 
# \begin{bmatrix}
#     k_{0,0} & k_{0,1} & \ldots & k_{0,14} \\
#     k_{1,0} & k_{1,1} & \ldots & k_{1,14}
# \end{bmatrix}
# \end{equation}
# $$
# 
# <br>
# 
# To later on solving the change of a surface species, one would simply obtain the sum of these reaction rates (multiplied with respective stoichiometric constant) for which it partakes.
# 
# <br>
# 
# $$
# \begin{equation}
#     \dfrac{\partial \theta_\text{H}}{\partial t} = 2 r_0 - r_1 - r_2 - r_3 - r_5 - r_6 - r_7 - r_9 - r_{11} - r_{12}
# \end{equation}
# $$
# 
# <br>
# 
# Manually writing a function describing this system could be quite a tedious task, especially as it would have to be redone everytime the definiton of the system changes. It would therefore be of interest to instead have a function which can construct this function based on a few parameters.
# 
# Given that we can express all the reactions and their rates as elementary steps, only the stoichiometery of the species is needed to construct an expression of this reaction system. For example, the rate expression $r_0$ and the change in $\theta_\text{H}$ evaluated for $r_0$ can be expressed as:
# 
# <br>
# 
# $$
# \begin{align}
#     r_0 &= k_{0,0} \, \theta_\text{∗}^{\nu_{*, 0}} \, p_{H2}^{\nu_{H2}, 0} - k_{1,0} \, \theta_H^{\nu_{H^*}, 0}\\
#     \left. \dfrac{\partial \theta_\text{H}}{\partial t} \right|_{r = r_0} &= \nu_{\text{H}, 0} \, r_0
# \end{align}
# $$
# 
# 
# <br>
# 
# As such, all we need is the stoichiometry, reaction constants and initial conditions in order to solve the problem. The variables we will look at belongs to two domains, the coverage by surface species ($x_\theta$) and the partial pressure of the gases ($x_p$), who together form the all the variables of the system ($x$).
# 
# <br>
# 
# $$
# \begin{equation}
# x_\theta = 
# \begin{bmatrix}
#     \theta_\text{∗} & \theta_\text{H} & \theta_\text{CHOO} & \theta_\text{CHOOH} & \theta_\text{CH$_2$OOH} & \theta_\text{OH} & \theta_\text{CH$_2$O} & \theta_\text{CH$_3$O} & \theta_\text{CO$_2$} & \theta_\text{COOH} & \theta_\text{CH$_2$OO} & \theta_\text{H$_2$O} & \theta_\text{CH$_3$OH}\\
# \end{bmatrix}
# \end{equation}
# $$
# 
# $$
# \begin{equation}
# x_p = 
# \begin{bmatrix}
#     p_\text{H$_2$} & p_\text{CO$_2$} & p_\text{CO} & p_\text{H$_2$O} & p_\text{CH$_3$OH} \\
# \end{bmatrix}
# \end{equation}
# $$
# 
# $$
# \begin{equation}
# x = 
# \begin{bmatrix}
#     x_\theta & x_p
# \end{bmatrix}
# \end{equation}
# $$
# 
# <br>
# 
# Based on equation (0) to (14), we can construct the following stoichiometry matrix ($S$), for which the rows correspond to $x$ and columns to $r$.
# 
# <br>
# 
# $$
# \begin{equation}
#     S =
#     %r_0 & r_1 & r_2 & r_3 & r_4 & r_5 & r_6 & r_7 & r_8 & r_9 & r_{10} & r_{11} & r_{12} & r_{13} & r_{14}\\
# %\begin{array}{c*{15}}
# \begin{bmatrix}
#     -2 &  0 &  1 &  1 & -1 &  1 &  1 &  1 & -1 &  1 &  0 &  1 &  1 &  1 &  1\\
#      2 & -1 & -1 & -1 &  0 & -1 & -1 & -1 &  0 & -1 &  0 & -1 & -1 &  0 &  0\\
#      0 &  1 & -1 &  0 &  0 &  0 &  0 &  0 &  0 &  0 &  0 & -1 &  0 &  0 &  0\\
#      0 &  0 &  1 & -1 &  0 &  0 &  0 &  0 &  0 &  0 &  0 &  0 &  0 &  0 &  0\\
#      0 &  0 &  0 &  1 & -1 &  0 &  0 &  0 &  0 &  0 &  0 &  0 &  1 &  0 &  0\\
#      0 &  0 &  0 &  0 &  1 & -1 &  0 &  0 &  0 &  0 &  1 &  0 &  0 &  0 &  0\\
#      0 &  0 &  0 &  0 &  1 &  0 & -1 &  0 &  0 &  0 &  0 &  0 &  0 &  0 &  0\\
#      0 &  0 &  0 &  0 &  0 &  0 &  1 & -1 &  0 &  0 &  0 &  0 &  0 &  0 &  0\\
#      0 &  0 &  0 &  0 &  0 &  0 &  0 &  0 &  1 & -1 &  0 &  0 &  0 &  0 &  0\\
#      0 &  0 &  0 &  0 &  0 &  0 &  0 &  0 &  0 &  1 & -1 &  0 &  0 &  0 &  0\\
#      0 &  0 &  0 &  0 &  0 &  0 &  0 &  0 &  0 &  0 &  0 &  1 & -1 &  0 &  0\\
#      0 &  0 &  0 &  0 &  0 &  1 &  0 &  0 &  0 &  0 &  0 &  0 &  0 & -1 &  0\\
#      0 &  0 &  0 &  0 &  0 &  0 &  0 &  1 &  0 &  0 &  0 &  0 &  0 &  0 & -1\\
#     -1 &  0 &  0 &  0 &  0 &  0 &  0 &  0 &  0 &  0 &  0 &  0 &  0 &  0 &  0\\
#      0 & -1 &  0 &  0 &  0 &  0 &  0 &  0 & -1 &  0 &  0 &  0 &  0 &  0 &  0\\
#      0 &  0 &  0 &  0 &  0 &  0 &  0 &  0 &  0 &  0 &  1 &  0 &  0 &  0 &  0\\
#      0 &  0 &  0 &  0 &  0 &  0 &  0 &  0 &  0 &  0 &  0 &  0 &  0 &  1 &  0\\
#      0 &  0 &  0 &  0 &  0 &  0 &  0 &  0 &  0 &  0 &  0 &  0 &  0 &  0 &  1\\
# \end{bmatrix}
# %\end{array}
# \end{equation}
# $$

# As this can be quite a labourious task for larger systems (with a high risk of making mistakes), we may instead opt for defining a function which can construct it for us. The input parameters to this function could then be a list declaring the variables of the system (and in which order we would like them listed in the resulting stoichiometry matrix) and a list of the reaction expressions. In the reaction expressions, we'll use `+` as the delimiter between the different species and `->` to separate the reactant and product. 

# In[1]:


vars_chem = ["*", "H*", "HCOO*", "HCOOH*", "CH2OOH*", "OH*", "CH2O*", 
             "CH3O*", "CO2*", "COOH*", "CH2OO*", "H2O*", "CH3OH*", 
             "H2 (g)", "CO2 (g)", "CO (g)", "H2O (g)", "CH3OH (g)"]

reactions = ["H2 (g) + 2* -> 2H*",
             "CO2 (g) + H* -> HCOO*",
             "HCOO* + H* -> HCOOH* + *",
             "HCOOH* + H* -> CH2OOH* + *",
             "CH2OOH* + * -> CH2O* + OH*",
             "OH* + H* -> H2O* + *",
             "CH2O* + H* -> CH3O* + *",
             "CH3O* + H* -> CH3OH* + *",
             "CO2 (g) + * -> CO2*",
             "CO2* + H* -> COOH* + *",
             "COOH* -> CO (g) + OH*",
             "HCOO* + H* -> CH2OO* + *",
             "CH2OO* + H* -> CH2OOH* + *",
             "H2O* -> H2O (g) + *",
             "CH3OH* -> CH3OH (g) + *",
            ]


# It then becomes a simple task of using regex in order to split the expression into its constituent variables and to check for their stoichiometry factor, as done for example, in the `constr_stoichio()` function below. In this case, the `constr_stoichio()` is only used to construct the stoichiometry matrix so information such as reversibility of the reaction is not of interest. However, if incorporated with another function which solves for instance reaction constants, then one would perhaps like to include additional parameters such as denoting `<->` for reversible reaction or such.

# In[2]:


import re

def constr_stoichio(lst_var, lst_react):
    stoichio_m = np.zeros((len(lst_var), len(lst_react)), dtype=np.int8)
    
    react_split = []
    for i, reaction in enumerate(lst_react):
        reaction = list(filter(None, re.split(r"\+| (->)", reaction)))
        reaction = [i.strip() for i in reaction]
        react_split.append(reaction) 
    
    for react_nr, reaction in enumerate(react_split):
        left_side = True
        
        for react_var in reaction:
            if react_var == "->":
                left_side = False
            else:
                try:
                    str_nu = re.match(r'\d+', react_var).group()
                    nu = int(str_nu)
                    react_var = react_var[len(str_nu):]
                except:
                    nu = 1

                if left_side == True:
                    nu = nu * -1

                try:
                    stoichio_idx = lst_var.index(react_var)
                    stoichio_m[stoichio_idx, react_nr] = nu
                except:
                    print(react_var 
                          + " doesn't exist in the list of variables.\n" \
                          + "Reaction " + str(react_nr) \
                          + " is potentially incorrect.\n" \
                          + str(lst_react[react_nr]) \
                          )
    
    return stoichio_m


# We might also want to check that the original expressions were correctly balanced. In the reaction system of this example, the number of surface sites remain constant. As such, we would expect to be able to solve $A x = 0$ for the surface species and sites, ergo a null space should exist for the part of the matrix composed of the surface species. We could then do the following test (making sure to exclude the gas species):

# In[3]:


import numpy as np
from scipy.linalg import null_space

calc_stoichio = constr_stoichio(vars_chem, reactions)
null_space(calc_stoichio[:-5,:]).any()


# So, here we have the class `BasicReactionModel` which we will use in order to solve our reaction system. In order to initialize an instance, we will need to supply reaction rate constants and a stoichiometry matrix (which we previously obtained from `constr_stoichio`). Note that as it is a class, we can easily setup several instances of reaction systems who are closed off from each other.
# 
# The concept of `BasicReactionModel` is as follows: the forwards and backwards reaction rates () are defined by equation $\text{(r.1)}$ and $\text{(r.2)}$, resulting in a vector, $\pmb{r}$, containing the net forward rate, equation$\text{(r.3)}$. A matrix multiplication between the stoichiometry matrix $\pmb{S}$ and the rate vector $\pmb{r}$ results in a matrix consisting of the differential for each variable in regards to every reaction step.
# 
# <br>
# 
# $$
# \begin{align}
# r_{i, \text{f}} &= k_{i, \text{f}} \underset{j=1}{\overset{n}{\prod}} \mid f(\nu_{i, j})\mid & 
# f(\nu_{i, j}) = \left\{
#     \begin{array}{ll}
#         x_i^{\mid \nu_{i, j} \mid } & \nu_{i, j} < 0 \\
#         1 & \text{else}
#     \end{array}
# \right. \tag{r.1}\\
# r_{i, \text{b}} &= k_{i, \text{b}} \underset{j=1}{\overset{n}{\prod}} \mid g(\nu_{i, j})\mid & 
# g(\nu_{i, j}) = \left\{
#     \begin{array}{ll}
#         x_i^{\mid \nu_{i, j} \mid } & \nu_{i, j} > 0 \\
#         1 & \text{else}
#     \end{array}
# \right. \tag{r.2}\\
# &\\
# \pmb{r} &= r_{\text{f}} - r_{\text{b}} \tag{r.3}\\
# &\\
# \dfrac{\partial \pmb{x}}{\partial t} &= \pmb{S} \, \pmb{r} \tag{r.4}
# \end{align}
# $$
# 
# <br>
# 
# While the stoichiometry matrix is conventionally written as above with variables in the row dimension and reactions in the column, `BasicReactionModel` instead utilises the transponate of this matrix in order to gain a more simple calculation.
# 
# To preserve constant terms (perhaps partial pressures) as constants, an additional vector `const_term` can be supplied to the class instance, declaring the variable as constant (`True`) or not (`False`). If `const_term` is given as $C(x_i)$, then one may express it as:
# 
# <br>
# 
# $$
# C(x_i) = \text{True} \Rightarrow \dfrac{\partial x_i}{\partial t} = 0
# $$

# In[4]:


class BasicReactionModel:
    """
    Constructs a reaction model to be used in microkinetic modelling.
    
    To define a reaction system, a stoichiometry matrix must be given during the
    initialisation of the class instance as well as a list of reaction rate 
    constants. If one or more variables are to be treated as constants, then the
    optional argument const_term must be given.
    
    Parameters
    ----------
    const_react : numpy array
      The constant terms (e.g reaction rate constant) of the forward (row 0) and 
      backward reaction (row 1) of the net reaction (col).
    stoichi : numpy array
      The stoichiometry matrix of the reaction system for the reactions 
      (row) and species (col).
    exp : numpy array
      The exponents of the reaction terms (col) at different reactions (row)
    const_term : numpy array
      1D Boolean array, defines which reaction terms that are set as 
      constants (True) and variables (False). Defaults to no constants. 
    forwM : numpy array
      2D Boolean array, signifying variables used for the forward reaction.
    backM : numpy array
      2D Boolean array, signifying variables used for the backward reaction.
    """

    def __init__(self,
                 const_react : np.array,
                 stoichi : np.array,
                 const_term={},
                 ):
        
        self.stoichio(stoichi)
        
        if const_term is self.__init__.__defaults__[0]:
            const_term = np.array([False] * self.stoichi.shape[1])
        
        if const_react.shape[1] != self.stoichi.shape[0]:
            if max(const_react.shape) == self.stoichi.shape[0]:
                const_react = const_react.T
            else:
                raise ValueError('Dimension mismatch between const_react                                   and stoichi')
        
        self.const = const_term
        self.react_const(const_react)
    
    def stoichio(self, stoichi):
        self.stoichi = stoichi.T
        self.exp = np.abs(self.stoichi[:])
        self.forwM = self.stoichi < 0
        self.backM = self.stoichi > 0
    
    def react_const(self, const_react):
        self.forwC = const_react[0, :]
        self.backC = const_react[1, :]

    def reaction(self, t, x):
        '''
        Solves dx/dt given the, for the class instance, predefined stoichiometry
        matrix and reaction rate constants.
        
        It is intended to be used together with an ODE solver such as SciPy's
        solve_ivp.        
        '''
        
        terms = (x * np.ones(self.stoichi.shape)) ** self.exp
        
        forw_r = self.forwC * np.prod(terms, where=self.forwM, axis=1)
        back_r = self.backC * np.prod(terms, where=self.backM, axis=1)

        dx = self.stoichi.T @ (np.abs(forw_r) - np.abs(back_r))
        dx[self.const] = 0

        return dx


# Given that we now have the `BasicReactionModel` class, `constr_stoichio` function and the lists of variables and reaction steps (`vars_chem` and `reactions`), we are now ready to define the reaction system.

# In[5]:


# Define starting conditions, setting 100% of sites as unoccupied
surf0 = [1] + [0] * (len(vars_chem) - 6)
# And pressures to some arbitrary pressure '1'
gas0 = [1] * 5
x0 = surf0 + gas0

# Some made up reaction rate constants, simply set to 1
k = np.array([[1] * len(reactions)]*2)

# Setting the formation of CO, H2O and CH3OH as irreversible by setting the
# reaction rate constants for their backwards reaction to 0
ir = ["CO (g)", "H2O (g)", "CH3OH (g)"]
ir_idx = [i for x in ir for i in range(len(reactions)) if x in reactions[i]]
k[1, ir_idx] = 0

# Likewise, we'll allow the surface species to vary in coverage but set the
# partial pressures of the gases as constant
constant_term = [False] * len(surf0) + [True] * len(gas0)

# Obtain the stoichiometry matrix
stoichi = constr_stoichio(vars_chem, reactions)

# Define the reaction system
TestModel = BasicReactionModel(k,
                               stoichi=calc_stoichio,
                               const_term=constant_term,
                               )


# Except the already imported numpy, we'll also need to import SciPy as we wish to use its initial value ODE solver `scipy.integrate.solve_ivp`. While we are at it, we'll also import matplotlib so that we can plot the results in the end.

# In[6]:


import scipy
from scipy.integrate import odeint

#%matplotlib widget
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# After defining the time span and error tolerance, we can then run the ODE solver. For this example, we'll use the default Runge-Kutta 45 solver.

# In[7]:


# Setting time span for the IVP ODE solver
t_span = [0, 10]

# ... and error tolerance
res = 1.0e-12
AbsTol = [res] * len(vars_chem)
RelTol = res

ans1 = scipy.integrate.solve_ivp(TestModel.reaction, 
                                 t_span, 
                                 x0, 
                                 rtol=RelTol, 
                                 atol=AbsTol,
                                 )


# While we could create a new instance if we would like to change a parameter of the model (for instance the reaction constants), we may also modify the existing one.
# 
# For this example, let's increase the rate constant of the dissociative adsorption of H$_2$ from the previous 1 to 10.

# In[8]:


# Let's change the rate constant for the dis. ads. of H2
k[0, 0] = 10
TestModel.react_const(k)

ans2 = scipy.integrate.solve_ivp(TestModel.reaction, 
                                 t_span, 
                                 x0,
                                 rtol=RelTol, 
                                 atol=AbsTol,
                                 )


# And we may now observe the change in reaction behaviour.

# In[9]:


plt.close(0)
fig = plt.figure(0)

ax1 = fig.add_subplot(2, 1, 1)
ax1.set_title('Original k')
ax1.set_ylabel('Coverage ' + r'$\theta$')
for i, name in enumerate(vars_chem[:-5]):
    ax1.plot(ans1.t, ans1.y[i], label=name)

ax2 = fig.add_subplot(2, 1, 2)
ax2.set_title('Changed k')
ax2.set_xlabel('Time')
ax2.set_ylabel('Coverage ' + r'$\theta$')
for i, name in enumerate(vars_chem[:-5]):
    ax2.plot(ans2.t, ans2.y[i], label=name)

h, a = ax1.get_legend_handles_labels()
ax1.legend(handles=h, loc="upper right", 
           ncol=int(np.ceil(len(vars_chem[:-5])/3)),
           frameon=False, fontsize=8
           )

fig.tight_layout();


# While this function works for surface reactions, additional work is needed in order to solve liquid and gas phase reactions as parameters such as total volume and volume change is not included. Furthermore, aspects such as mass transfer is also not accounted for. While one could redefine the `reaction` function of `BasicReactionModel`, an alternative is to define additional functions which modifies the input/output in order to account for these additional factors. 
# 
# As an example, let's define the function `stim_sin` which allows us to periodically vary one of the parameters using a sinus function.

# In[10]:


def stim_sin(t, x, func, const):
    '''
    Varies one or more input arguments to a function func with a sinus function.
    
    Its intended use is to add a stimulus function to a function to be solved by
    ODE solver.
    
    func : function
        function describing the ODE system. func(t, x)
    const : list
        The coefficient and constant term of the sinus function.
        x_stim[0] = x[0] * np.sin(const[0, 0] * t + const[0, 1])
        If const[i, 0] = 0, then x_stim[i] = x[i]
    '''
    
    x_stim = [term[0] * np.sin(term[1] * t + term[2]) + x[i] 
              if term[0] != 0 
              else x[i] 
              for i, term in enumerate(const)]
    
    return func(t, x_stim)


# Let us now see how the system behaves if the H$_2$ feed were perturbed by this sinus function.

# In[11]:


period = 1
mod_gas = "H2 (g)"
idx_mod_gas = vars_chem.index(mod_gas)
stim_const = np.array([[.0] * 3] * len(x0))
stim_const[idx_mod_gas, 0] = x0[vars_chem.index(mod_gas)] / 2
stim_const[idx_mod_gas, 1] = 2 * np.pi * (1 / period)

ans3 = scipy.integrate.solve_ivp(stim_sin, 
                                 t_span, 
                                 x0,
                                 args=[TestModel.reaction, stim_const],
                                 rtol=RelTol, 
                                 atol=AbsTol,
                                 )

plt.close(1)
fig = plt.figure(1)

ax = fig.add_subplot(1, 1, 1)
ax.set_title(r'Disturbed H$_2$ feed')
ax.set_ylabel('Coverage ' + r'$\theta$')
ax.set_xlabel('Time')
for i, name in enumerate(vars_chem[:-5]):
    ax.plot(ans3.t, ans3.y[i], label=name)

h, a = ax.get_legend_handles_labels()
ax.legend(handles=h, loc="upper right", 
          ncol=int(np.ceil(len(vars_chem[:-5])/3)),
          frameon=False, fontsize=8
          )

fig.tight_layout();

