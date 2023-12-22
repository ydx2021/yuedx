"""Artistic License 2.0
Copyright (c) [2023], [Dongxiao Yue]

Preamble

This license establishes the terms under which a given free software Package may be copied, modified, distributed, and/or redistributed. The intent is that the Copyright Holder maintains some artistic control over the development of that Package while still keeping the Package available as open source and free software.
You are always permitted to make arrangements wholly outside of this license directly with the Copyright Holder of a given Package. If the terms of this license do not permit the full use that you propose to make of the Package, you should contact the Copyright Holder and seek a different licensing arrangement.

Definitions

"Copyright Holder" means the individual(s) or organization(s) named in the copyright notice for the entire Package.

"Contributor" means any party that has contributed code or other material to the Package, in accordance with the Copyright Holder's procedures.

"You" and "your" means any person who would like to copy, distribute, or modify the Package.

"Package" means the collection of files distributed by the Copyright Holder, and derivatives of that collection and/or of those files. A given Package may consist of either the Standard Version, or a Modified Version.

"Distribute" means providing a copy of the Package or making it accessible to anyone else, or in the case of a company or organization, to others outside of your company or organization.

"Distributor Fee" means any fee that you charge for Distributing this Package or providing support for this Package to another party. It does not mean licensing fees.

"Standard Version" refers to the Package if it has not been modified, or has been modified only in ways explicitly requested by the Copyright Holder.

"Modified Version" means the Package, if it has been changed, and such changes were not explicitly requested by the Copyright Holder.

"Original License" means this Artistic License as Distributed with the Standard Version of the Package, in its current version or as it may be modified by The Perl Foundation in the future.

"Source" form means the source code, documentation source, and configuration files for the Package.
​
"Compiled" form means the compiled bytecode, object code, binary, or any other form resulting from mechanical transformation or translation of the Source form.

Permission for Use and Modification Without Distribution

(1) You are permitted to use the Standard Version and create and use Modified Versions for any purpose without restriction, provided that you do not Distribute the Modified Version.

Permissions for Redistribution of the Standard Version

(2) You may Distribute verbatim copies of the Source form of the Standard Version of this Package in any medium without restriction, either gratis or for a Distributor Fee, provided that you duplicate all of the original copyright notices and associated disclaimers. At your discretion, such verbatim copies may or may not include a Compiled form of the Package.
​
(3) You may apply any bug fixes, portability changes, and other modifications made available from the Copyright Holder. The resulting Package will still be considered the Standard Version, and as such will be subject to the Original License.

Distribution of Modified Versions of the Package as Source

(4) You may Distribute your Modified Version as Source (either gratis or for a Distributor Fee, and with or without a Compiled form of the Modified Version) provided that you clearly document how it differs from the Standard Version, including, but not limited to, documenting any non-standard features, executables, or modules, and provided that you do at least ONE of the following:
     (a) make the Modified Version available to the Copyright Holder of the Standard Version, under the Original License, so that the Copyright Holder may
          include your modifications in the Standard Version.
     (b) ensure that installation of your Modified Version does not prevent the user installing or running the Standard Version. In addition, the Modified 
          Version must bear a name that is different from the name of the Standard Version.
     (c) allow anyone who receives a copy of the Modified Version to make the Source form of the Modified Version available to others under
          (i) the Original License or
          (ii) a license that permits the licensee to freely copy, modify and redistribute the Modified Version using the same licensing terms that apply to the
               copy that the licensee received, and requires that the Source form of the Modified Version, and of any works derived from it, be made freely
               available in that license fees are prohibited but Distributor Fees are allowed.

Distribution of Compiled Forms of the Standard Version or Modified Versions without the Source

(5) You may Distribute Compiled forms of the Standard Version without the Source, provided that you include complete instructions on how to get the Source of the Standard Version. Such instructions must be valid at the time of your distribution. If these instructions, at any time while you are carrying out such distribution, become invalid, you must provide new instructions on demand or cease further distribution. If you provide valid instructions or cease distribution within thirty days after you become aware that the instructions are invalid, then you do not forfeit any of your rights under this license.

(6) You may Distribute a Modified Version in Compiled form without the Source, provided that you comply with Section 4 with respect to the Source of the Modified Version.
​
Aggregating or Linking the Package

(7) You may aggregate the Package (either the Standard Version or Modified Version) with other packages and Distribute the resulting aggregation provided that you do not charge a licensing fee for the Package. Distributor Fees are permitted, and licensing fees for other components in the aggregation are permitted. The terms of this license apply to the use and Distribution of the Standard or Modified Versions as included in the aggregation.
(8) You are permitted to link Modified and Standard Versions with other works, to embed the Package in a larger work of your own, or to build stand-alone binary or bytecode versions of applications that include the Package, and Distribute the result without restriction, provided the result does not expose a direct interface to the Package.

Items That are Not Considered Part of a Modified Version

(9) Works (including, but not limited to, modules and scripts) that merely extend or make use of the Package, do not, by themselves, cause the Package to be a Modified Version. In addition, such works are not considered parts of the Package itself, and are not subject to the terms of this license.

General Provisions
(10) Any use, modification, and distribution of the Standard or Modified Versions is governed by this Artistic License. By using, modifying or distributing the Package, you accept this license. Do not use, modify, or distribute the Package, if you do not accept this license.

(11) If your Modified Version has been derived from a Modified Version made by someone other than you, you are nevertheless required to ensure that your Modified Version complies with the requirements of this license.

(12) This license does not grant you the right to use any trademark, service mark, tradename, or logo of the Copyright Holder.

(13) This license includes the non-exclusive, worldwide, free-of-charge patent license to make, have made, use, offer to sell, sell, import and otherwise transfer the Package with respect to any patent claims licensable by the Copyright Holder that are necessarily infringed by the Package. If you institute patent litigation (including a cross-claim or counterclaim) against any party alleging that the Package constitutes direct or contributory patent infringement, then this Artistic License to you shall terminate on the date that such litigation is filed.

(14) Disclaimer of Warranty: THE PACKAGE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS "AS IS' AND WITHOUT ANY EXPRESS OR IMPLIED WARRANTIES. THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR NON-INFRINGEMENT ARE DISCLAIMED TO THE EXTENT PERMITTED BY YOUR LOCAL LAW. UNLESS REQUIRED BY LAW, NO COPYRIGHT HOLDER OR CONTRIBUTOR WILL BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING IN ANY WAY OUT OF THE USE OF THE PACKAGE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

import math
import sys

import matplotlib.pyplot as plt
import mpmath
import numpy as np
from matplotlib.ticker import FuncFormatter
from scipy import integrate


def erf_approx_abramowitz_stegun(x):
    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429

    t = 1.0 / (1.0 + p * x)
    poly = a1 * t + a2 * t**2 + a3 * t**3 + a4 * t**4 + a5 * t**5
    return 1.0 - poly * np.exp(-x**2)


def erfc_Karagiannidis_Lioumpas(x):
     A, B = 1.98,1.135
     return (1- np.exp(-A*x)) *np.exp(-x*x)/B/np.sqrt(np.pi)/x
    
    
    
mpmath.mp.dps = 60

#erf_func = erf_approx_abramowitz_stegun
#erfc_func = erfc_Karagiannidis_Lioumpas
erf_func = mpmath.erf 
erfc_func = mpmath.erfc
exp_func = mpmath.exp

#the Y function in the paper

def yuedx_integration(y1, y2, z, lv, ERF_ARG_LIMIT=15, COSZ_LIMIT=15, useErf=False):
    if(y1<0):
        print(f"y1<0: {y1}")
        return 0
    if(y2==0):
        return 0 
    
    if(z > np.pi/2):
       print(f"zenith angle > pi/2, you need to change the starting point of integration, see paper")
       return 0
        
    sin_z = np.sin(z)
    erf_arg = np.sqrt(lv * (1 - sin_z + y2))
    erf_arg0 = np.sqrt(lv * (1 - sin_z + y1))
    term_common = 1/ (np.sqrt(lv*(1 + sin_z)))
    term_1 = -erf_arg * np.exp(-lv * y2)
    term_10 = -erf_arg0 ** np.exp(-lv * y1)
    term_2_common = np.sqrt(np.pi) * (lv * sin_z + 1/2)
    
    if erf_arg0 > ERF_ARG_LIMIT:
        if COSZ_LIMIT >= ERF_ARG_LIMIT and erf_arg0 > COSZ_LIMIT:
            return (lv+0.5)/(lv*np.cos(z))
        term_2_fac = 1/np.sqrt(np.pi*lv) *(np.exp(-lv*y1)/np.sqrt(1-sin_z+y1) - np.exp(-lv*y2)/np.sqrt(1-sin_z+y2))
    else:
        if(useErf):
                term_2_fac = exp_func(-lv *(sin_z-1))  *(erf_func(erf_arg) - erf_func(erf_arg0))
        else:
                term_2_fac = exp_func(-lv *(sin_z-1))  *(-erfc_func(erf_arg) + erfc_func(erf_arg0))
   
    return term_common * ( (term_1 -term_10)*1 + term_2_common*term_2_fac)
    
    

EARTH_RADIUS = 6.4e6
def calc_scale_height_boltzmann(radius_meters=EARTH_RADIUS, density_g_cm=5.51, air_molecule_atomic_weight=30, temperature_kelvin=300):
    G_const = 6.67e-11
    R_ideal = 8.31 # K_B*N_A
    g = np.pi*4/3 * radius_meters*G_const*density_g_cm*1e3
    scale_height  = R_ideal * temperature_kelvin/air_molecule_atomic_weight*1e3/g
    return scale_height, radius_meters/scale_height

    

def calc_ys_from_x(x, z, start_height):
    
    if z >= np.pi/2:
        y1 = np.sqrt(1+x*x- 2*np.cos(z)*x) -1 
        return [start_height, z, y1 ]
    

erf_integrand_lambda = lambda x: np.exp (- x*x)*2/np.sqrt(np.pi)

def erf_diff_integral (x1, x2):
    return integrate.quad(erf_integrand_lambda, x1, x2)

def compare_asymptotic_diff(lv, sin_z, y):
    erf_arg = np.sqrt(lv * (1 - sin_z + y))
    erf_arg0 = np.sqrt(lv * (1 - sin_z ))
    asymptotic_diff = 1/np.sqrt(np.pi*lv)*np.exp(-lv*(1-sin_z)) *(1/np.sqrt(1-sin_z) - np.exp(-lv*y)/np.sqrt(1-sin_z+y))
    numerical_diff, err = erf_diff_integral(erf_arg0, erf_arg)
    dif_pct = (asymptotic_diff-numerical_diff)/numerical_diff*100
    erfc_1 = erfc_func(erf_arg)
    erfc_0 = erfc_func(erf_arg0)
    mpmath_erfc0 = mpmath.nstr(erfc_0, 12)
    mpmath_erf_diff = mpmath.nstr(erfc_0 - erfc_1, 12)
    print (f"arg={erf_arg0},erfc_0={mpmath_erfc0}, input={(lv, sin_z, y)} asymptotic={asymptotic_diff}, integral={numerical_diff}, ErfDiff={mpmath_erf_diff}, diff%={dif_pct:.2f}%")
        

#compare_asymptotic_diff(600, 0.1, 0.3)
    
#h=calc_scale_height_boltzmann(radius_meters=EARTH_RADIUS/3)
#print(h)
#sys.exit(0)


Rv = 6.4e6 #earth radius
Atmos = 0.1 # ratio of atmosphere depth to radius
StartingHeight = 0.005 # starting height relative to Height scale
icount = 45


def y_to_x (y, zv):
    return np.cos(zv) + np.sqrt((y+1)**2-np.sin(zv)**2)

def degree_formatter(x, pos):
    return f"{int(x)}°"

def chapman_integral(lv, zv):
    champman_integrand = lambda u, lvv, zvv: np.exp (- np.sqrt( lvv*lvv+u**2+2*lvv*u*np.cos(zvv)) +lvv)
    return integrate.quad(champman_integrand, 0, np.inf, args=(lv, zv), epsabs=1.49e-08)


def plot_yuedx_v_y(ymax, N, lv):
    # Generate N evenly spaced theta values from 0 to 90 degrees
    
    theta_values = np.linspace(0, 90, N)
    
    # x values from 0 to 1
    x_values = np.linspace(0, ymax, 100)  # 100 points between 0 and 1
    
    # Plot y vs x for each theta
    plt.figure(figsize=(10, 6))
    for theta in theta_values:
        tv = theta *np.pi/180
        y_values = [yuedx_integration(0, x, tv, lv) for x in x_values]  # Calculate y for each x
        plt.plot(x_values, y_values, label=f'z = {theta:.1f}°')
    
    plt.title(f'Y vs y for Different Zenith Angles ($\lambda$={lv})')
    plt.xlabel('y')
    plt.ylabel('Y')
    plt.xscale('log')  # Set y-axis to log scale
   
    plt.legend(loc='upper left', bbox_to_anchor=(0.7, 0.9)) 
    plt.grid(True)
    plt.show()

def compare_yintegration_vs_numerical(lv_vals=[0.1], ASYMPTO_SWITCH=7):
    data_points_by_scale={}
    data_points_by_lambda={}
    
    
    integrand_lambda = lambda x, lvv, Zv: lvv*np.exp (- lvv *(np.sqrt( 1+x**2+2*np.cos(Zv)*x) -1))

    for lvv in lv_vals:
        Dv = Rv*(1+1/lvv/2) # start position
        Rd = Rv/Dv
           
        y0 = 0
        y1 = 1/10 #Atmos*Rv/Dv
        
        x0_values = []
        x1_values = []
        analytical_values = []
        analytical_values_asymptotic = []
        numerical_values = []
        chapman_values = []
        dif_values = []  # Percent differences for the first comparison
        dif_values_asymptotic = [] 
        dif_chapman_values = []  # Percent differences for the first comparison
        z_values=[]
        
        start = 0
        
        for i in range(icount+1):
            
            enda = 90
            zva = start+ (enda-start)*i/icount
            zv = zva *np.pi/180
            
            x0=0 
            x1 = y_to_x(y1, zv)
            
            numerical_integral, error = integrate.quad(integrand_lambda, x0, x1, args=(lvv, zv), epsabs=1.49e-09,)
            numerical_values.append(numerical_integral)
            #print(f"numer={numerical_integral:.4f}, err={error:.4f}, z={tva:.1f}, Re={Rev:.2f}")
            
           
            analytical_value = yuedx_integration(y0, y1, zv, lvv,ERF_ARG_LIMIT=200, useErf=False)
            analytical_values.append(analytical_value)
            
    
            dif = (analytical_value-numerical_integral)/(numerical_integral +1e-2)*100
            dif_values.append(dif)
    
            
            #print(f"prox1={d2_numer:.4f}, dif={dif:.3f}%, z={tva:.1f} ")
            
            if (abs(dif) > 10):
                print(f" dif={dif}, lambda={lvv}, z={zv*180/3.14} ")
            
            analytical_value_a = yuedx_integration(y0, y1, zv, lvv, ERF_ARG_LIMIT=ASYMPTO_SWITCH, COSZ_LIMIT=ASYMPTO_SWITCH+1,useErf=False)
            analytical_values_asymptotic.append(analytical_value_a)
            
    
            dif = (analytical_value_a-numerical_integral)/(numerical_integral +1e-2)*100
            dif_values_asymptotic.append(dif)
    
            
            #print(f"prox1={d2_numer:.4f}, dif={dif:.3f}%, z={tva:.1f} ")
            
            if (abs(dif) > 10):
                print(f" dif={dif}, lambda={lvv}, z={zv*180/3.14} ")
                
            chapman_v, error = chapman_integral(lvv, zv)
            chapman_values.append(chapman_v)
            
            
            dif = (analytical_value-chapman_v)/(chapman_v +1e-2)*100
            dif_chapman_values.append(dif)
            if (abs(dif) > 10):
                print (f"dif={dif},  lambda={lvv}, z={zv*180/3.14}  champam={chapman_v},{error} analytical={analytical_value}")
            
            x0_values.append(x0)
            x1_values.append(x1)  # Assuming x1 is defined in your calculations
            z_values.append(zv*180/np.pi)
       

        # data_points_by_scale[ScaleHeight]= [z_values, analytical_values, numerical_values, dif_values]
        data_points_by_lambda[lvv]= [z_values, analytical_values, numerical_values, dif_values,analytical_values_asymptotic, dif_values_asymptotic, chapman_values, dif_chapman_values]
    

    
    figdim = (12, 8)
   
    # Plotting Analytical vs Numerical Integral for Multiple lambda values
    sorted_ls = sorted(data_points_by_lambda.keys())
    plt.figure(figsize=figdim)  # Create new figure
    for lv in sorted_ls:
        z_values, approx1_values, numerical_values, diffs, analytic_vals_a, analytic_diffs_a, chapman_vals, chapman_diffs   = data_points_by_lambda[lv]
        plt.plot(z_values, approx1_values, label=f'Analytical ($\lambda$={lv:.1f})')
        plt.scatter(z_values, numerical_values, label=f'Numerical ($\lambda$={lv:.1f})')
    plt.title(f'Analytical vs Numerical Integral at Different $\lambda$ Values (y={y1})')
    plt.xlabel('Zenith angle (Degrees)')
    plt.xlim(0, 90)  # Set x-axis to end at 90
    plt.ylabel('Integral Value')
    plt.yscale('log')  # Set y-axis to log scale
    plt.gca().xaxis.set_major_formatter(FuncFormatter(degree_formatter))
    plt.legend()
    plt.xticks(np.arange(min(z_values), max(z_values)+1, 10))
    plt.show()
    
    # Plotting Differences for lambda values
    plt.figure(figsize=figdim)  # Create new figure
    for lv in sorted_ls:
        z_values, approx1_values, numerical_values, diffs, analytic_vals_a, analytic_diffs_a, chapman_vals, chapman_diffs   = data_points_by_lambda[lv]
        plt.plot(z_values, diffs, label=f'$\lambda$={lv:.1f}')
    plt.title(f'Relative Error (%)  Analytical vs. Numerical Integral (y={y1})')
    plt.xlabel('Zenith angle (Degrees)')
    plt.xlim(0, 90)  # Set x-axis to end at 90
    plt.ylabel('Relative Error (%)')
    plt.gca().xaxis.set_major_formatter(FuncFormatter(degree_formatter))
    plt.legend()
    plt.xticks(np.arange(min(z_values), max(z_values)+1, 10))
    plt.show()
   
    plt.figure(figsize=figdim)  # Create new figure
    for lv in sorted_ls:
        z_values, approx1_values, numerical_values, diffs, analytic_vals_a, analytic_diffs_a, chapman_vals, chapman_diffs   = data_points_by_lambda[lv]
        plt.plot(z_values, analytic_diffs_a, label=f'$\lambda$={lv:.1f}')
    plt.title(f'Relative Error (%) Asymptotic Approximation vs. Numerical (y={y1})')
    plt.xlabel('Zenith angle (Degrees)')
    plt.xlim(0, 90)  # Set x-axis to end at 90
    plt.ylabel('Difference (%)')
    plt.gca().xaxis.set_major_formatter(FuncFormatter(degree_formatter))
    plt.legend()
    plt.xticks(np.arange(min(z_values), max(z_values)+1, 10))
    plt.show()
    
    
    plt.figure(figsize=figdim)  # Create new figure
    for lv in sorted_ls:
        z_values, approx1_values, numerical_values, diffs, analytic_vals_a, analytic_diffs_a, chapman_vals, chapman_diffs   = data_points_by_lambda[lv]
        plt.plot(z_values, approx1_values, label=f'Analytical ($\lambda$={lv:.1f})')
        plt.scatter(z_values, chapman_vals, label=f'Chapman ($\lambda$={lv:.1f})')
    plt.title(f'Analytical vs Chapman Function at Different $\lambda$ Values (y={y1})')
    plt.xlabel('Zenith angle (Degrees)')
    plt.xlim(0, 90)  # Set x-axis to end at 90
    plt.ylabel('Integral Value')
    plt.yscale('log')  # Set y-axis to log scale
    plt.gca().xaxis.set_major_formatter(FuncFormatter(degree_formatter))
    plt.legend()
    plt.xticks(np.arange(min(z_values), max(z_values)+1, 10))
    plt.show()
    
    plt.figure(figsize=figdim)  # Create new figure
    for lv in sorted_ls:
        z_values, approx1_values, numerical_values, diffs, analytic_vals_a, analytic_diffs_a, chapman_vals, chapman_diffs   = data_points_by_lambda[lv]
        plt.plot(z_values, chapman_diffs, label=f'$\lambda$={lv:.1f}')
    plt.title(f'Difference (%)  Analytical and Chapman Function (y={y1})')
    plt.xlabel('Zenith angle (Degrees)')
    plt.xlim(0, 90)  # Set x-axis to end at 90
    plt.ylabel('Difference (%)')
    plt.gca().xaxis.set_major_formatter(FuncFormatter(degree_formatter))
    plt.legend()
    plt.xticks(np.arange(min(z_values), max(z_values)+1, 10))
    plt.show()
    
    

compare_yintegration_vs_numerical(lv_vals=[50, 100, 200, 500, 1000,10000], ASYMPTO_SWITCH=12)

plot_yuedx_v_y(0.1,9, 100)

#sys.exit(0)
#print(d_taylor_x_int)
