#importing
import os
import numpy as np
from astropy.io import fits as FITS
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math

#creating dictionary to store data with cycles as key
cycle=["pks1510_1","pks1510_2","pks1510_3","pks1510_4","pks1510_5","pks1510_6","pks1510_7","pks1510_8","pks1510_9","pks1510_10"]
data={}
for i in cycle:
    path=os.listdir(f"/home/gnanachandru/PycharmProjects/FirstProject/Data/{i}")
    data[i]=[]
    for file in path:
        if ".spec_" in file:
            data[i].append(file)

#checking our dictionary
for key in data:
    print(len(data[key]),key,data[key])

#functions
#to plot the data
def view_data(path):
    hdu=FITS.open(path)
    X_data=np.linspace(4014e-10,8138e-10,len(hdu[0].data))
    Y_data=hdu[0].data
    plt.figure(figsize=(20,10))
    plt.loglog(X_data,Y_data)
    # plt.show()
    plt.close

#to get a and b values
def a_and_b(i):  #function to get a and b for different x values
    if 0.3<=i<=1.1:
        a=0.574*(i**1.61)
        b=-0.527*(i**1.61)
    elif 1.1<=i<=3.3:
        y=i-1.82
        a=1+0.17699*y-0.50447*y**2-0.02427*y**3+0.72085*y**4+0.01979*y**5-0.77530*y**6+0.32999*y**7
        b=1.41338*y+2.28305*y**2+1.07233*y**3-5.38434*y**4-0.62251*y**5+5.30260*y**6-2.90002*y**7
    return a,b

#for reddening correction
def reddening_correction(a,b):
    result=0.088*(a*3.1+b)      #we get E(B-V) value from Nasa Extragalactic Database
    return result

#fit function
def fit_func(x,a,b,c):
    return a*np.power(abs(x),b)+c

#gaussian fit function
def fit_gaussian(x,a,m,s):
    return a*np.exp(-((x-m)**2)/(2*s**2))

# temperature calculation
def temp_calc(fwhm, mean_lambda):
    k_B = 1.38e-23
    m = 1.67e-27
    c = 3.0e8
    result = (m / (2 * k_B)) * (((c * fwhm) / mean_lambda) ** 2)
    return result

#velocity calc
def velocity_calc(fwhm,mean_lambda):
    result=(3.0e8*fwhm)/mean_lambda
    return result

def equivalent_width(start,end):
    dlambda=(8000e-07-4000e-07)/len(X_data)
    mask1=((merged_data[:,0]>start) & (merged_data[:,0]<end))
    masked_data1=merged_data[mask1]
    X_masked1=masked_data1[:,0]
    Y_masked1=masked_data1[:,1]
    f_lambda=Y_masked1
    f_continoum=fit_func(X_masked1,a_opt,b_opt,c_opt)
    e_width=0
    for i in range(len(X_masked1)):
        e_width += (1 - (f_lambda[i] / f_continoum[i])) * dlambda
    return e_width

def chi_square(Y_data,Y_fit,sigma):
    N=len(Y_data)
    p=3
    chi_sq=np.sum(((Y_data-Y_fit)/sigma)**2)/(N-p)
    return chi_sq

def log_likelyhood(Y_data,Y_fit,sigma):
    residuals=Y_data-Y_fit
    logL=-0.5*np.sum((residuals)/sigma)**2 + np.log(2*np.pi*sigma**2)
    return logL




FWHM=[]
TEMPERATURE=[]
VELOCITY=[]
EQUIVALENT_WIDTH_1=[]
EQUIVALENT_WIDTH_2=[]
EQUIVALENT_WIDTH_3=[]
EQUIVALENT_WIDTH_4=[]
EQUIVALENT_WIDTH_5=[]
EQUIVALENT_WIDTH_6=[]
EQUIVALENT_WIDTH_7=[]
A_OPT=[]
B_OPT=[]
C_OPT=[]
REDUCED_CHI=[]
LOG_LIKELYHOOD=[]

#calculation for all the data
for cycle in data:
    for img in data[cycle]:
        path=f"/home/gnanachandru/PycharmProjects/FirstProject/Data/{cycle}/{img}"
        hdu = FITS.open(path)
        X_data = np.linspace(4014e-10, 8138e-10, len(hdu[0].data))
        Y_data = hdu[0].data
        # plt.figure(figsize=(20, 10))
        # plt.loglog(X_data, Y_data)
        # plt.show()
        # plt.close

        #reddening correction
        x=1/(X_data*(10**6))
        Y_data_corrected=[]
        a_lambda=[]
        for i in x:
            a,b=a_and_b(i)
            al=reddening_correction(a,b)
            a_lambda.append(al)
        #calculating the corrected flux
        for i in range(len(x)):
            Y_data_corrected.append(Y_data[i]*10**(0.4*a_lambda[i]))

        plt.figure(figsize=(20,10))
        plt.subplot(1,2,1)
        plt.title("raw")
        plt.loglog(X_data,Y_data)

        plt.subplot(1,2,2)
        plt.loglog(X_data,Y_data_corrected)
        plt.title("corrected")
        plt.plot([5.5e-07],[6.57e-16],marker="o",color="r")
        # plt.show()
        plt.close()

        merged_data=np.column_stack((X_data,Y_data_corrected))
        mask=((merged_data[:,0]<4.0564e-07) | (merged_data[:,0]>4.42992e-07)) & \
             ((merged_data[:,0]<5.7698e-07) | (merged_data[:,0]>6.00096e-07)) & \
             ((merged_data[:,0]<6.1018e-07) | (merged_data[:,0]>6.36139e-07)) & \
             ((merged_data[:,0]<6.36139e-07) | (merged_data[:,0]>6.79988e-07)) & \
             ((merged_data[:,0]<6.93063e-07) | (merged_data[:,0]>7.32e-07)) & \
             ((merged_data[:,0]<7.32943e-07) | (merged_data[:,0]>7.55082e-07))

        masked_data=merged_data[mask]
        X_masked=masked_data[:,0]
        Y_masked=masked_data[:,1]

        #powerlaw fit
        popt,pcov=curve_fit(fit_func,X_masked,Y_masked,p0 = [1e-15, -2, 1e-15],maxfev=50000)
        a_opt,b_opt,c_opt=popt
        A_OPT.append(a_opt)
        B_OPT.append(b_opt)
        C_OPT.append(c_opt)

        #to compare the fit and graph
        plt.figure(figsize=(20,10))
        plt.loglog(X_data,Y_data_corrected,c="g")
        plt.loglog(X_masked,fit_func(X_masked,a_opt,b_opt,c_opt))
        # plt.show()
        plt.close()

        #plot after subtracting fit from data
        plt.figure(figsize=(20,10))
        Y_data_peak=Y_data_corrected-fit_func(X_data,a_opt,b_opt,c_opt)
        plt.loglog(X_data,Y_data_peak)
        # plt.show()
        plt.close()

        mask1=((merged_data[:,0]>6.93e-07) & (merged_data[:,0]<7.2e-07))
        masked_data1=merged_data[mask1]
        X_masked1=masked_data1[:,0]
        Y_masked1=masked_data1[:,1]

        #plotting the prominent emission only for further calulations
        plt.figure(figsize=(20,10))
        Y_data_peak1=Y_masked1-fit_func(X_masked1,a_opt,b_opt,c_opt)
        plt.loglog(X_masked1,Y_data_peak1)
        # plt.show()
        plt.close()

        #guessing the parameters
        a_guess=max(Y_masked1)
        m_guess=X_masked1[np.argmax(Y_masked1)]





        #guassian fit
        popt1,pcov1=curve_fit(fit_gaussian,X_masked1,Y_masked1,p0=[a_guess,m_guess,0.077e-07],maxfev=50000)
        a1_opt,m1_opt,s1_opt=popt1
        plt.figure(figsize=(20,10))
        plt.loglog(X_masked1,fit_gaussian(X_masked1,a1_opt,m1_opt,s1_opt),c="g")
        plt.loglog(X_masked1,Y_data_peak1)
        # plt.show()
        plt.close()

        fwhm=2.355*s1_opt
        FWHM.append(fwhm)

        t1=temp_calc(fwhm,m1_opt)
        v1=velocity_calc(fwhm,m1_opt)
        e1=equivalent_width(4.05e-07,4.42e-07)
        e2 = equivalent_width(5.76e-07, 6.0e-07)
        e3 = equivalent_width(6.1e-07, 6.36e-07)
        e4=equivalent_width(6.36e-07,6.79e-07)
        e5 = equivalent_width(6.93e-07, 7.2e-07)
        e6=equivalent_width(7.2e-07,7.32e-07)
        e7 = equivalent_width(7.32e-07, 7.55e-07)
        TEMPERATURE.append(t1)
        VELOCITY.append(v1)
        EQUIVALENT_WIDTH_1.append(e1)
        EQUIVALENT_WIDTH_2.append(e2)
        EQUIVALENT_WIDTH_3.append(e3)
        EQUIVALENT_WIDTH_4.append(e4)
        EQUIVALENT_WIDTH_5.append(e5)
        EQUIVALENT_WIDTH_6.append(e6)
        EQUIVALENT_WIDTH_7.append(e7)

        reduced_chi_square=chi_square(Y_masked1,fit_gaussian(X_masked1,a1_opt,m1_opt,s1_opt),s1_opt)
        log_l=log_likelyhood(Y_masked1,fit_gaussian(X_masked1,a1_opt,m1_opt,s1_opt),s1_opt)

        REDUCED_CHI.append(reduced_chi_square)
        LOG_LIKELYHOOD.append(log_l)




print(np.mean(A_OPT),np.mean(B_OPT),np.mean(C_OPT))
print(np.mean(FWHM))
print(len(TEMPERATURE),np.mean(TEMPERATURE),TEMPERATURE)
print(len(VELOCITY),np.mean(VELOCITY),VELOCITY)
print(len(EQUIVALENT_WIDTH_1),np.mean(EQUIVALENT_WIDTH_1),EQUIVALENT_WIDTH_1)
print(len(EQUIVALENT_WIDTH_2),np.mean(EQUIVALENT_WIDTH_2),EQUIVALENT_WIDTH_2)
print(len(EQUIVALENT_WIDTH_3),np.mean(EQUIVALENT_WIDTH_3),EQUIVALENT_WIDTH_3)
print(len(EQUIVALENT_WIDTH_4),np.mean(EQUIVALENT_WIDTH_4),EQUIVALENT_WIDTH_4)
print(len(EQUIVALENT_WIDTH_5),np.mean(EQUIVALENT_WIDTH_5),EQUIVALENT_WIDTH_5)
print(len(EQUIVALENT_WIDTH_6),np.mean(EQUIVALENT_WIDTH_6),EQUIVALENT_WIDTH_6)
print(len(EQUIVALENT_WIDTH_7),np.mean(EQUIVALENT_WIDTH_7),EQUIVALENT_WIDTH_7)
print(np.mean(REDUCED_CHI))
print(np.mean(LOG_LIKELYHOOD))








































