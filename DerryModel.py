#%matplotlib inline
from scipy.integrate import ode
import numpy as np
from scipy.interpolate import PchipInterpolator as pchip
from matplotlib import pyplot as plt

class DerryModel(object):
    """A near clone of Lou Derry's Matlab 3-box model in Python.

       Using an object here to make the semantics a little more Pythonic...
       (I.e. we are initializing stuff once and only once,
       we are evaluating derivatives in their own routine, etc.)
    """


    # For the initializer, we are using Python syntax for a named argument, with a default value.
    # To change anything from its default, simply pass that as a named argument with a different value.
    # E.g. foo = DerryModel(rain=0.30) will set rain to 0.30 and leave everything else at defaults.
    def __init__(self,
                 wflux = 0.001,       # 10^18 m3/yr, surf-deep water exchang rate
                 vold = 1.23,         # 10^18 m3, volume of deep ocean
                 vols = 0.12,         # 10^18 m3, volume of surface ocean
                 prod = 0.000175,     # 10^18 mol/yr, export production
                 rain = 0.25,         # rain ratio, carbonate/Corg
                 distime = 8.64,      # years, dissolution time scale for air-sea ex
                 watemp = 288.,       # surface ocean T, K
                 matmco2 = 0.0495 ):  # 10^18 mol, 1 PAL = 280 ppmv
        """Initialized everything once and only once..."""
        self.wflux = wflux
        self.vold = vold
        self.vols = vols
        self.prod = prod
        self.rain = rain
        self.distime = distime
        self.watemp = watemp
        self.matmco2 = matmco2

        zeroC = 273.15

        # This was originally 278.0 in the Matlab code. Lou thinks it was a probable typo on his part.
        # FIXME: check similarity of answers with older value...
        self.T_ref = zeroC

        #%%%%%%%%%%%%%%
        #land uptake model
        #landcoeff = .0;


        #derived parameters
        self.kcarb = 0.000575 + 0.000006*(self.watemp - self.T_ref)   # note that "kcarb" here = K2/K1
        self.kco2 = 0.035 + 0.0019*(self.watemp - self.T_ref)         # Henry's Law  as as f(temp)

        # Initialize the emissions profile.
        self.emissions(scenario='A2')
        # Initial Values
        self.pco2 = 1.               #normalized to preindustrial pCO2 = 280 ppm
        self.sigcs = 2.0248          #mol/m3 (sigma-CO2 surface)
        self.sigcd = 2.24723         #mol/m3 (sigma-CO2 deep)
        self.alks = 2.19886          #eq/m3 (surface alkalinity)
        self.alkd = 2.26011          #eq/m3 (deep alkalinity)
        return
    
    def emissions(self,scenario='A2'):
        """We are grabbing the historical and appending the chosen future scenario CO2 emissions profiles here.
        
        The historical emissions are cut and paste from Lou's CDIAC10_history.xslx spreadsheet
        (part of this git repository).
        
        The future scenarios are from an IPCC report. (FIXME: Lou? Citation?)
        """
        
        self.EMISS_HIST = np.array([         [0,6.67E-07],
        [10,8.33E-07],
        [20,1.17E-06],
        [30,2.00E-06],
        [40,2.75E-06],
        [50,4.50E-06],
        [60,7.58E-06],
        [70,1.23E-05],
        [80,1.97E-05],
        [90,2.97E-05],
        [100,4.45E-05],
        [110,6.83E-05],
        [120,7.77E-05],
        [130,8.78E-05],
        [140,1.08E-04],
        [150,1.36E-04],
        [160,2.15E-04],
        [170,3.40E-04],
        [180,4.43E-04],
        [190,5.12E-04],
        [200,5.62E-04],
        [210,7.64E-04]],dtype=np.float32)
        
        #Future emissions scenarios from MiniCAM A1, A2, B1, B2 at 10 yr intervals
        #Lay them out row-wise because it's easier to type. Then transpose 'em into column vectors to match the HIST.
        emis_A1=np.array([[220., 230., 240., 250., 260., 270., 280., 290., 300.],\
                          [1.03E-03, 1.26E-03, 1.45E-03, 1.60E-03,\
                           1.64E-03, 1.69E-03, 1.75E-03, 1.54E-03, 1.33E-03]],dtype=np.float32).T
        emis_A2=np.array([[220., 230., 240., 250., 260., 270., 280., 290., 300.],\
                          [9.03E-04, 1.02E-03, 1.16E-03, 1.35E-03,\
                           1.54E-03, 1.74E-03, 1.94E-03, 2.18E-03, 2.44E-03]],dtype=np.float32).T
        emis_B1=np.array([[220., 230., 240., 250., 260., 270., 280., 290., 300.],\
                          [7.94E-04, 8.44E-04, 8.28E-04, 7.87E-04,\
                           7.45E-04, 6.73E-04, 5.73E-04, 5.06E-04, 4.40E-04]],dtype=np.float32).T
        emis_B2=np.array([[220., 230., 240., 250., 260., 270., 280., 290., 300.],\
                          [8.74E-04, 9.59E-04, 1.02E-03, 1.10E-03,\
                           1.15E-03, 1.20E-03, 1.22E-03, 1.23E-03, 1.23E-03]],dtype=np.float32).T
        
        if(scenario == 'A1'):
            self.emis = np.concatenate([self.EMISS_HIST,emis_A1])
        elif(scenario == 'A2'):
            self.emis = np.concatenate([self.EMISS_HIST,emis_A2])
        elif(scenario == 'B1'):
            self.emis = np.concatenate([self.EMISS_HIST,emis_B1])
        elif(scenario == 'B2'):
            self.emis = np.concatenate([self.EMISS_HIST,emis_B2])
        else:
            raise ValueError('Unknown future scenario name')

        #Initialize pchip interpolation on t as specifed by ode solver
        # Matlab's call is structured yi = pchip(x,y,xi)
        # where x,y are vectors containing the "control points" (independent and dependent variables, in sequence)
        # xi is a vector containing "the independent variable values at which the interpolation should be evaluated"
        # and the return yi is a vector containing the interpolated dependent variable's values at the xi locations.
        # The original Matlab call is:
        # fuel=pchip(emis(:,1), emis(:,2), t)
        # In words, I think this is saying that x,y are the columns of emis, t is vector of times at which to 
        # evaluate things for the ode solver -- computed in some fashion yet to be determined,
        # and fuel is the corresponding values from the interpolation
        #
        # Mapping that to Python's implementation of pchip (aliased in the import above)
        # we need to initialize the interpolator with the control points (only need to do this once)
        # and then call the function at each value of t to return fuel(t).
        # We initialize here, and stash the initialized function away as a variable (essentially a function pointer)
        # in self to be used during the ode solution stuff...
        self.interpolate_emissions = pchip(self.emis[:,0],self.emis[:,1])


    def walkeralk(self,t,y):
        """function ydot = walkeralk(t,y)"""

        # build empty solution vector each call
        # Used internally, then garbage collected
        # N.B. (5,) looks like a typo, but isn't. 
        # It's Python syntax for a single element tuple.
        ydot = np.zeros((5,),dtype=np.float32)


    
        # WARNING! Unlike Matlab (or FORTRAN for that matter) 
        # Python uses zero based instead of one based indexing!
        # variable list:
        # y[0] = pCO2, y[1] = sigcs, y[2] = sigcd, y[3] = alks, y[4] = alkd

        #carbonate species
        hco3 = y[1]-np.sqrt(y[1]**2 - y[3]*(2.*y[1] - y[3])*(1.-4.*self.kcarb))/(1.-4.*self.kcarb)
        co3 = (y[3] - hco3)/2.

        hco3s = hco3
        co3s = co3
        pco2s = self.kco2*(hco3s**2)/co3s


    
        # The interpolation. 
        # We've already initialized the function with the control points in __init__.
        # Now we just need to call it at t.
        fuel=self.interpolate_emissions(t)


        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #The basic differential equations

        # variable list:
        # Same as above...

        ydot[0] = (pco2s-y[0])/self.distime + 0.8*fuel/self.matmco2 #- landcoeff*fuel       #pCO2
        ydot[1] = (-(pco2s - y[0])/self.distime*self.matmco2                    
                   - (1. + self.rain)*self.prod + (y[2] - y[1])*self.wflux)/self.vols       #sigcs

        ydot[2] = ((1. + self.rain)*self.prod - (y[2] - y[1])*self.wflux)/self.vold         #sigcd
        ydot[3] = ((y[4] - y[3])*self.wflux - (2.*self.rain - 0.15)*self.prod)/self.vols    #alks
        ydot[4] = ((2.*self.rain - 0.15)*self.prod - (y[4] - y[3])*self.wflux)/self.vold    #alkd
    
        return ydot

    def walker(self, Vm = 2.905661):
        """The top-level function to be called to run the model."""
        # define initial conditions
        y0 =np.array([self.pco2, self.sigcs, self.sigcd, self.alks, self.alkd],dtype=np.float32)
        
        #set integration time limits
        tfinal = 300.
        #note: start year is 1800 CE.
        tspan = [0., tfinal];

        # evaluate diffeq function using ode45 routine
        # <http://www.mathworks.com/help/matlab/ref/ode45.html#bti6n8p-40>
        # Matlab's ode45 is based on an explicit Runge-Kutta (4,5) formula, the Dormand-Prince pair.
        #[t,y] = ode45(@walkeralk,tspan,y0);
        #
        # As mentioned above, dopri5 is the SciPy equivalent R-K integrator scheme.
        # The Matlab code (as called by Lou's original version) appears 
        # to return values at every integration timestep.
        # The SciPy code can be coerced into doing that, but it's simpler to 
        # use it to integrate to a specified time value and report that as output.
        # We'll use that form here, with the expectation that since the underlying numerics is the same
        # all that _should_ happen (in a perfect world) is that our explicit time steps get inserted
        # into the integration scheme with little harm done to the solution. Fingers crossed.
        #
        # So, let's evaluate the solution every year for the tfinal year duration of the run.
        dt = 1.0

        self.r = ode(self.walkeralk).set_integrator('dopri5')
        self.r.set_initial_value(y0,tspan[0])

        # Let's keep the answers, shall we? ;-)

        z = (tspan[1]-tspan[0])/dt
        t = np.zeros(z,dtype=np.float32)
        # y is vector valued
        y = np.zeros((z,5),dtype=np.float32)
        i = 0
        # Begin our explicit time stepping
        while self.r.successful() and self.r.t < tspan[1]:
            self.r.integrate(self.r.t+dt)
            t[i] = self.r.t
            y[i,:] = self.r.y
            i += 1
        #assert i == (z-1)

        # Now the derived variables

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        watemp = 288.        #surface ocean T, Kelvins


        k1 =10**(-6.10)         #K1 equilibrium constant H2CO3 -> HCO3- + H+
        k2 =10**(-9.33)         #K2 equilibrium constant HCO3- -> CO3= + H+

        #derived parameters from fitting equations to constants (Walker)
        kcarb = 0.000575 + 0.000006*(watemp - self.T_ref) #note that "kcarb" here = K2/K1

        #carbonate species
        hco3 = y[:,1]-np.sqrt(y[:,1]**2 - y[:,3]*(2.*y[:,1] - y[:,3])*(1.-4.*kcarb))/(1.-4.*kcarb)
        co3 = (y[:,3] - hco3)/2.

        #update surface values
        hco3s = hco3
        co3s = co3
        acid = (k2*hco3s/co3s)
        pH= -np.log10(acid)
        #convert pCO2 to ppm for plotting
        CO2_ppm = 280. * y[:,0]
        #generate calendar time scale for diffeq output
        calentime=t+1800.

        #Berkeley temp model

        alpha = 8.342105       #baseline T, ?C

        #CO2 forcing log-linear
        beta = 4.466369        #CO2 coefficient
        CO2o = 277.3           #baseline CO2, ppm

        #volcanic forcing
        # Now passed in as an argument as a test of the interactive slider stuff
        #Vm = 2.905661         #mean 20th cent volcanic forcing from Berkely Earth project
                               #obtained by averaging the 12 month moving average data from 1900 thru 1999

        gamma = -0.01515       #volcanic aerosol coefficient

        # Matlab's syntax for ln is log, just like Python's...
        T_model = alpha + beta*(np.log(CO2_ppm/277.3)) + gamma*Vm

        # We've already initialized the interpolator.
        # Let's turn it into a ufunc and the use it...
        # Slight trickery here. Equivalent to looping over all entries...
        array_interp_emiss = np.frompyfunc(self.interpolate_emissions,1,1)
        emisp=array_interp_emiss(t)
        x=len(emisp)

        emissions = emisp*1000.*44.  #convert 10^18 mol CO2/yr to Gton CO2

        #generate output file for later plotting
        B2_results = np.array([calentime, emissions, CO2_ppm, pH])
        np.savetxt('B2_out.txt', B2_results, delimiter=', ', 
                    header='calentime, emissions, CO2_ppm, pH')
        #xlswrite('B2scenarios.xls',B2_results)
        #xlswrite('CDIAC10_history', EMIS_HIST)
        #plot results

        fig1 = plt.figure(1,figsize=(18.0,6.0))

        plt.subplot(231)
        plt.plot(calentime,emissions,'r',linewidth=1.5)
        plt.title('CO2 emissions, Gt/yr')
        plt.xlim(1800,2100)
        plt.ylim(0,110)

        plt.subplot(232)
        plt.plot(calentime,CO2_ppm,'m',linewidth=1.5)
        plt.title('pCO2, ppmv')
        plt.xlim([1800, 2100])
        plt.ylim([250, 1100])

        plt.subplot(233)
        plt.plot(calentime,y[:,1],'c',linewidth=1.5)
        plt.xlim([1800, 2100])
        plt.ylim([2, 2.4])
        plt.plot(calentime,y[:,2],'k',linewidth=1.5)
        plt.title('TCO2')
        plt.ylabel('mmol/kg')
        plt.legend(('TCO2surf', 'TCO2deep'), loc='upper left')

        plt.subplot(234)
        plt.plot(calentime, pH, linewidth=1.5)
        plt.title('surface ocean pH')
        plt.xlim([1800, 2100])
        plt.ylim([7.8, 8.4])

        plt.subplot(235)
        plt.plot(calentime,hco3, 'r',linewidth=1.5)
        plt.title('surface HCO3-, mM')
        plt.xlim([1800, 2100])
        plt.ylim([1.8, 2.2]);

        plt.subplot(236);
        plt.plot(calentime,co3, 'g',linewidth=1.5)
        plt.title('surface CO3=, mM')
        plt.xlim([1800, 2100])
        plt.ylim([0.05, 0.25])

        fig1.savefig('fig1.png', dpi=fig1.dpi)


        # No fig. 2jQuery20301879276212672375_1405302322683?
        fig3 = plt.figure(3,figsize=(18.0,6.0))
        plt.subplot(211)
        plt.plot(calentime, CO2_ppm, 'b',linewidth=1.5)
        plt.title('CO2 ppm')
        plt.xlim([1800, 2100])
        plt.ylim([200, 1200])

        plt.subplot(212)
        plt.plot(calentime, T_model, 'r',linewidth=1.5)
        plt.title('temperature C')
        plt.xlim([1800, 2100])
        plt.ylim([8, 15])
        fig3.savefig('fig3.png', dpi=fig3.dpi)

        return

if __name__ == '__main__':
    model = DerryModel()
    model.walker(Vm = 2.905661)