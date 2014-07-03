   %three box model of carbon system from Walker 1991 
%   "Numerical Adventures with Geochemical Cycles", ch. 5.
% boxes are atmosphere, surface ocean, and deep ocean
% exchange between the atm and surf box depends on the pCO2 difference
% and a kinetic constant "distime (dissolution time scale).
% exchange between the surf and deep boxes is given by a mass exchange term
% "wflux".
% A productivity term ("prod") controls the biological pump for sigma CO2
% ("sigco2") and  alkalinity  ("alk").
% algebraic solutions for HCO3- ("hco3"), CO3= ("co3") and pH ("pH") are computed from standard
% eqns.



function walker
%close all               %useful to close figure panes between runs
clear variables

load CDIAC10r;      %load CDIAC 10 yr interval emissions, units 10^18 mol/yr

%initial values
pco2 = 1;               %normalized to preindutrial pCO2 = 280 ppm
sigcs = 2.0248;         %mol/m3 (sigma-CO2 surface)
sigcd = 2.24723;        %mol/m3 (sigma-CO2 deep)
alks = 2.19886;         %eq/m3 (surface alkalinity)
alkd = 2.26011;         %eq/m3 (deep alkalinity)

%define initial condition 
y0 =[pco2 sigcs sigcd alks alkd]';

%set integration time limits
tfinal = 300;
%note: start year is 1800 CE. 
tspan = [0, tfinal];

%evaluate diffeq function using ode45 routine
[t,y] = ode45(@walkeralk,tspan,y0);
z=size(t)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
watemp = 288;        %surface ocean T, Kelvins 


k1 =10^-6.10;       %K1 equilibrium constant H2CO3 -> HCO3- + H+
k2 =10^-9.33;       %K2 equilibrium constant HCO3- -> CO3= + H+

%derived parameters from fitting equations to constants (Walker) 
kcarb = 0.000575 + 0.000006*(watemp - 278); %note that "kcarb" here = K2/K1

%carbonate species
hco3 = y(:,2)-sqrt(y(:,2).^2 - y(:,4).*(2*y(:,2) - y(:,4))*(1-4*kcarb))/(1-4*kcarb);
co3 = (y(:,4) - hco3)/2;

%update surface values
hco3s = hco3;
co3s = co3;
acid = (k2*hco3s./co3s);
pH=-log10(acid);
%convert pCO2 to ppm for plotting
CO2_ppm = 280*y(:,1);
%generate calendar time scale for diffeq output
calentime=t+1800;

%Berkeley temp model

alpha = 8.342105;       %baseline T, ?C

%CO2 forcing log-linear
beta = 4.466369;        %CO2 coefficient
CO2o = 277.3;           %baseline CO2, ppm

%volcanic forcing
Vm = 2.905661    %mean 20th cent volcanic forcing from Berkely Earth project
%obtained by averaging the 12 month moving average data from 1900 thru 1999

gamma = -0.01515;     %volcanic aerosol coefficient

T_model = alpha + beta*(log(CO2_ppm/277.3)) + gamma*Vm;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Parameterize fossil fuel emissions data:
%10 yr interval data in 1e18 mol/yr from CDIAC, originally given as Mg C/yr
global EMIS_HIST
EMIS_HIST=CDIAC10r';        %CDIAC historical emissions data 1800-2010

%Future emissions scenarions from MiniCAM models A1, A2, B1, B2 at 10 yr
%intervals
emis_A1=[220 230 240 250 260 270 280 290 300; ...
    1.03E-03 1.26E-03 1.45E-03 1.60E-03	1.64E-03 1.69E-03 1.75E-03 1.54E-03 1.33E-03]';
emis_A2=[220 230 240 250 260 270 280 290 300; ...
    9.03E-04 1.02E-03 1.16E-03 1.35E-03	1.54E-03 1.74E-03 1.94E-03 2.18E-03	2.44E-03]';
emis_B1=[220 230 240 250 260 270 280 290 300; ...
   7.94E-04	8.44E-04 8.28E-04 7.87E-04 7.45E-04 6.73E-04 5.73E-04 5.06E-04 4.40E-04]';
emis_B2=[220 230 240 250 260 270 280 290 300; ...
    8.74E-04 9.59E-04 1.02E-03 1.10E-03	1.15E-03 1.20E-03 1.22E-03 1.23E-03 1.23E-03]';

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%concatenate historical and projected emissions to get full 300 yr vector
emis = cat(1, EMIS_HIST, emis_A2);
%need to change this line and also line 182 to run different scenarios
%%%%%%%%%%%%%%%%%%%%%%%%%%%

%implement pchip interpolation on t as specifed by ode solver
emisp=pchip(emis(:,1), emis(:,2), t);
x=size(emisp);

emissions = emisp*1000*44;  %convert 10^18 mol CO2/yr to Gton CO2 

%generate output file for later plotting
B2_results = [calentime emissions CO2_ppm pH];
save B2_out.txt B2_results -ascii
%xlswrite('B2scenarios.xls',B2_results)
%xlswrite('CDIAC10_history', EMIS_HIST)
%plot results
clf

figure(1)

subplot(231);
plot(calentime,emissions,'r','LineWidth',1.5); 
title('CO2 emissions, Gt/yr'), axis([1800 2100 0 110]);  

subplot(232);
plot(calentime,CO2_ppm,'m','LineWidth',1.5); 
title('pCO2, ppmv'), axis([1800 2100 250 1100]);  

subplot(233);
plot(calentime,y(:,2),'c','LineWidth',1.5), axis([1800 2100 2 2.4]); 
hold on; plot(calentime,y(:,3),'k','LineWidth',1.5); 
title('TCO2'), ylabel('mmol/kg');
legend('TCO2surf', 'TCO2deep', 'Location', 'Northwest');
hold off;

subplot(234);   
plot(calentime, pH, 'LineWidth',1.5), title('surface ocean pH'), axis([1800 2100 7.8 8.4]);

subplot(235);
plot(calentime,hco3, 'r','LineWidth',1.5), title('surface HCO3-, mM'), axis([1800 2100 1.8 2.2]);

subplot(236);
plot(calentime,co3, 'g','LineWidth',1.5), title('surface CO3=, mM'), axis([1800 2100 0.05 0.25]);

figure(3)
subplot(211);
plot(calentime, CO2_ppm, 'b','LineWidth',1.5), title('CO2 ppm'), axis([1800 2100 200 1200]);
subplot(212);
plot(calentime, T_model, 'r','LineWidth',1.5), title('temperature C'), axis([1800 2100 8 15]);

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% this section defines the differential equations and constants solved above

function ydot = walkeralk(t,y)

%build empty solution vector
ydot = zeros(5,1);
 
%parameters
wflux = 0.001;       %10^18 m3/yr, surf-deep water exchang rate
vold = 1.23;         %10^18 m3, volume of deep ocean
vols = 0.12;         %10^18 m3, volume of surface ocean
prod = 0.000175;     %10^18 mol/yr, export production
rain = 0.25;         %rain ratio, carbonate/Corg
distime = 8.64;      %years, dissolution time scale for air-sea ex
watemp = 288;        %surface ocean T, K 
matmco2 = 0.0495;    %10^18 mol, 1 PAL = 280 ppmv

%%%%%%%%%%%%%%%
%land uptake model
%landcoeff = .0;


%derived parameters
kcarb = 0.000575 + 0.000006*(watemp - 278); %note that "kcarb" here
                                            % = K2/K1
kco2 = 0.035 + 0.0019*(watemp - 278);   %Henry's Law  as as f(temp)

%carbonate species
hco3 = y(2)-sqrt(y(2)^2 - y(4)*(2*y(2) - y(4))*(1-4*kcarb))/(1-4*kcarb);
co3 = (y(4) - hco3)/2;

hco3s = hco3;
co3s = co3;
pco2s = kco2*hco3s^2/co3s;


%historical fossil fuel emissions from CDIAC in EMIST_HIST
global EMIS_HIST


%Future emissions scenarios from MiniCAM A1, A2, B1, B2 at 10 yr intervals
emis_A1=[220 230 240 250 260 270 280 290 300; ...
1.03E-03 1.26E-03 1.45E-03 1.60E-03	1.64E-03 1.69E-03 1.75E-03 1.54E-03 1.33E-03]';
emis_A2=[220 230 240 250 260 270 280 290 300; ...
    9.03E-04 1.02E-03 1.16E-03 1.35E-03	1.54E-03 1.74E-03 1.94E-03 2.18E-03	2.44E-03]';
emis_B1=[220 230 240 250 260 270 280 290 300; ...
   7.94E-04	8.44E-04 8.28E-04 7.87E-04 7.45E-04 6.73E-04 5.73E-04 5.06E-04 4.40E-04]';
emis_B2=[220 230 240 250 260 270 280 290 300; ...
    8.74E-04 9.59E-04 1.02E-03 1.10E-03	1.15E-03 1.20E-03 1.22E-03 1.23E-03 1.23E-03]';


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%concatenate historical and projected emissions to get full 300 yr vector
emis = cat(1, EMIS_HIST, emis_A2);
%implement pchip interpolation on t as specifed by ode solver
fuel=pchip(emis(:,1), emis(:,2), t);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%The basic differential equations

%variable list:
%y(1) = pCO2, y(2) = sigcs, y(3) = sigcd, y(4) = alks, y(5) = alkd

ydot(1) = (pco2s-y(1))/distime+0.8*fuel/matmco2; %- landcoeff*fuel;       %pCO2
ydot(2) = (-(pco2s - y(1))/distime*matmco2 ...
    - (1 + rain)*prod + (y(3) - y(2))*wflux)/vols;     %sigcs
    
ydot(3) = ((1 + rain)*prod - (y(3) - y(2))*wflux)/vold;   %sigcd
ydot(4) = ((y(5) - y(4))*wflux - (2*rain - 0.15)*prod)/vols; %alks
ydot(5) = ((2*rain - 0.15)*prod - (y(5) - y(4))*wflux)/vold;  %alkd

return