# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 14:06:42 2020

@author: Saumya Dholakia
"""
#Input parameters
import numpy as np
L=1
a=1
n=1
tend= (L/a)/2
dx= 0.025
x = np.arange(0,L+dx,dx)
zeta = x-a*tend
nx = np.size(x)

#Creating dictionaries
clist = dict()
clist[0.1] = 0.1
clist[0.5] = 0.5
clist[0.9] = 0.9
clist[1.0] =1.0
colors = dict()
colors[0.1]='red'
colors[0.5]='blue'
colors[0.9]='green'
colors[1.0]='yellow'

#PART 1:LF and LW methods sine function
#The Lax-Fredrich algorithm
class lax_f():
    def solve(self,x,uinitial,c,a,tend):
        import numpy as np
        dx = x[1]-x[0]
        dt = c*dx/a
        nx = np.size(x)
        uold = np.zeros(nx)
        unew = np.zeros(nx)
        t = np.arange(0,tend,dt)
        uold = np.copy(uinitial)
        for j in t:
            for i in range(1,nx-1):
                unew[i]=(uold[i+1]+uold[i-1])/2 -c*(uold[i+1]-uold[i-1])/2
            unew[0]=(uold[1]+uold[nx-2])/2 -c*(uold[1]-uold[nx-2])/2
            unew[nx-1] = unew[0]
            for i in range(nx):
                uold[i]=unew[i]
        return unew
    
#Initial condition and Exact solution for the sine function
uinitial = (1. + np.sin(2.*np.pi*n*x))/2
uexact = (1. + np.sin(2.*np.pi*n*zeta))/2

#Solver 1 LF method
solver1 = lax_f()
u_LF_sine = dict()
for c in clist: 
    u_LF_sine[c] = solver1.solve(x,uinitial,c,a,tend)

#Plots - u-x plot,uinitial-x plot and uexact-x plot   
import matplotlib.pyplot as plt
f, ax = plt.subplots(1,1,figsize=(8,5))
ax.plot(x,uinitial,label='Initial condition',color='black',)
ax.plot(x,uexact,label='Exact solution',color='black', linestyle='--')
for c in clist:
    ax.plot(x,u_LF_sine[c],label='c='+str(c),color=colors[c])
ax.set_xlabel('$x$',size=20)
ax.set_ylabel('$u(x,t)$',size=20)
ax.grid()
ax.legend(fontsize=16)
plt.savefig('LF_sine.png',bbox_inches='tight')

#Error-x plot (LF Sine error)
f, ax = plt.subplots(1,1,figsize=(8,5))
for c in clist:
    ax.plot(x,u_LF_sine[c]-uexact,label='c='+str(c),color=colors[c]) 
ax.set_xlabel('$x$',size=20)
ax.set_ylabel('$\epsilon = u-u_{exact}$',size=20)
ax.grid()
ax.legend(fontsize=16)
plt.savefig('LF_sine_error.png',bbox_inches='tight')

#The Lax-Werendoff algorithm
class lax_w():
    def solve(self,x,uinitial,c,a,tend):
        import numpy as np
        dx = x[1]-x[0]
        dt = c*dx/a
        nx = np.size(x) 
        uold = np.zeros(nx)
        unew = np.zeros(nx)
        t = np.arange(0,tend,dt)
        uold = np.copy(uinitial)
        for j in t:
            for i in range(1,nx-1):
                unew[i] =uold[i]-c*(uold[i+1]-uold[i-1])/2 +(c*c)*(uold[i+1]-2*uold[i]+uold[i-1])/2
            unew[0] =uold[0]-c*(uold[1]-uold[nx-2])/2 +(c*c)*(uold[1]-2*uold[0]+uold[nx-2])/2
            unew[nx-1] = unew[0]
            for i in range(nx):
                uold[i]=unew[i]
        return unew

#Solver 2 LW method (Sine and Top hat)
solver2 = lax_w()
u_LW_sine = dict()
uinitial = (1. + np.sin(2.*np.pi*n*x))/2
for c in clist: 
    u_LW_sine[c] = solver2.solve(x,uinitial,c,a,tend)
    uexact = (1. + np.sin(2.*np.pi*n*zeta))/2

#Plots - u-x plot,uinitial-x plot and uexact-x plot (Sine function)
f, ax = plt.subplots(1,1,figsize=(8,5))
ax.plot(x,uinitial,label='Initial condition',color='black',)
ax.plot(x,uexact,label='Exact solution',color='black', linestyle='--')
for c in clist:
    ax.plot(x,u_LW_sine[c],label='c='+str(c),color=colors[c]) 
ax.set_xlabel('$x$',size=20)
ax.set_ylabel('$u(x,t)$',size=20)
ax.grid()
ax.legend(fontsize=16)
plt.savefig('LW_sine.png',bbox_inches='tight')

#Error-x plot (LW sine error)
f, ax = plt.subplots(1,1,figsize=(8,5))
for c in clist:
    ax.plot(x,u_LW_sine[c]-uexact,label='c='+str(c),color=colors[c]) 
ax.set_xlabel('$x$',size=20)
ax.set_ylabel('$\epsilon = u-u_{exact}$',size=20)
ax.grid()
ax.legend(fontsize=16)
plt.savefig('LW_sine_error.png',bbox_inches='tight')

#PART 2:LF and LW methods top hat function
#Initial condition and Exact solution for the sine function
uinitial = np.heaviside(x,1.)-np.heaviside(x-L/4,1.)
uexact = np.heaviside(zeta,1.)-np.heaviside(zeta-L/4,1.)

#Solver 1 LF method (Sine and Top hat)
solver1 = lax_f()
u_LF_top_hat = dict()
for c in clist: 
    u_LF_top_hat[c] = solver1.solve(x,uinitial,c,a,tend)
    
#Plots - u-x plot,uinitial-x plot and uexact-x plot (Top hat)
f, ax = plt.subplots(1,1,figsize=(8,5))
ax.plot(x,uinitial,label='Initial condition',color='black',)
ax.plot(x,uexact,label='Exact solution',color='black', linestyle='--')
for c in clist:
    ax.plot(x,u_LF_top_hat[c],label='c='+str(c),color=colors[c])
ax.set_xlabel('$x$',size=20)
ax.set_ylabel('$u(x,t)$',size=20)
ax.grid()
ax.legend(fontsize=16)
plt.savefig('LF_Top hat.png',bbox_inches='tight')

#Error-x plot (Top hat)
f, ax = plt.subplots(1,1,figsize=(8,5))
for c in clist:
    ax.plot(x,u_LF_top_hat[c]-uexact,label='c='+str(c),color=colors[c])
ax.set_xlabel('$x$',size=20)
ax.set_ylabel('$\epsilon = u-u_{exact}$',size=20)
ax.grid()
ax.legend(fontsize=16)
plt.savefig('LF_top_hat_error.png',bbox_inches='tight')

#Solver 2 LW method (Sine and Top hat)
solver2 = lax_w()
u_LW_top_hat = dict()
for c in clist: 
    u_LW_top_hat[c] = solver2.solve(x,uinitial,c,a,tend)
    
#Plots - u-x plot,uinitial-x plot and uexact-x plot (Top hat) - LW method
f, ax = plt.subplots(1,1,figsize=(8,5))
ax.plot(x,uinitial,label='Initial condition',color='black',)
ax.plot(x,uexact,label='Exact solution',color='black', linestyle='--')
for c in clist:
    ax.plot(x,u_LW_top_hat[c],label='c='+str(c),color=colors[c])
ax.set_xlabel('$x$',size=20)
ax.set_ylabel('$u(x,t)$',size=20)
ax.grid()
ax.legend(fontsize=16)
plt.savefig('LF_Top hat.png',bbox_inches='tight')

#Error-x plot (Top hat) - LW method
f, ax = plt.subplots(1,1,figsize=(8,5))
for c in clist:
    ax.plot(x,u_LW_top_hat[c]-uexact,label='c='+str(c),color=colors[c])
ax.set_xlabel('$x$',size=20)
ax.set_ylabel('$\epsilon = u-u_{exact}$',size=20)
ax.grid()
ax.legend(fontsize=16)
plt.savefig('LW_top_hat_error.png',bbox_inches='tight')

#PART3: MDE error terms for LF and LW methods sine function
#The Lax-Fredrich algorithm - First error term for the MDE
class error_lax_f1():
    def solve(self,x,uinitial,c,a,tend):
        import numpy as np
        dx = x[1]-x[0]
        dt = c*dx/a
        nx = np.size(x)
        uold = np.zeros(nx)
        uxx = np.zeros(nx)
        t = np.arange(0,tend,dt)
        uold = np.copy(uinitial)
        for j in t:
            for i in range(1,nx-1):
                uxx[i] =(a*dx/2)*((1/c) -c)*(uold[i+1]-2*uold[i]+uold[i-1])/(dx*dx)
            uxx[0]=(a*dx/2)*((1/c) -c)*(uold[1]-2*uold[0]+uold[nx-2])/(dx*dx)
            uxx[nx-1] = uxx[0]
        return uxx
    
#Initial condition and Exact solution for the sine function
uinitial = (1. + np.sin(2.*np.pi*n*x))/2
uexact = (1. + np.sin(2.*np.pi*n*zeta))/2

# Solver 3 LF method MDE error term 1 (Sine function)
solver3 = error_lax_f1()
Lfmde1=dict()
for c in clist:
    Lfmde1[c]=solver3.solve(x,uinitial,c,a,tend)

#Plot: MDE error term 1 - LF method (Sine function)
f, ax = plt.subplots(1,1,figsize=(8,5))
for c in clist:
    ax.plot(x,Lfmde1[c],label='c='+str(c),color=colors[c])    
ax.set_xlabel('$x$',size=20)
ax.set_ylabel('$LF-MDE-First-term$',size=20)
ax.grid()
ax.legend(fontsize=16)
plt.savefig('LF MDE1.png',bbox_inches='tight')

#The Lax-Fredrich algorithm - Second error term for the MDE
class error_lax_f2():
    def solve(self,x,uinitial,c,a,tend):
        import numpy as np
        dx = x[1]-x[0]
        dt = c*dx/a
        nx = np.size(x)
        uold = np.zeros(nx)
        uxxx = np.zeros(nx)
        t = np.arange(0,tend,dt)
        uold = np.copy(uinitial)
        for j in t:
            for i in range(2,nx-2):
                uxxx[i] =(a*dx*dx/3)*((1) -c*c)*(uold[i+2]+2*uold[i-1]-2*uold[i+1] -uold[i-2])/(2*dx*dx*dx)
            uxxx[1] =(a*dx*dx/3)*((1) -c*c)*(uold[3]+2*uold[0]-2*uold[2] -uold[nx-2])/(2*dx*dx*dx)
            uxxx[0] =(a*dx*dx/3)*((1) -c*c)*(uold[2]+2*uold[nx-2]-2*uold[1] -uold[nx-3])/(2*dx*dx*dx)
            uxxx[nx-2] =(a*dx*dx/3)*((1) -c*c)*(uold[1]+2*uold[nx-3]-2*uold[nx-1] -uold[nx-4])/(2*dx*dx*dx)
            uxxx[nx-1] = uxxx[0]
        return uxxx 

# Solver 4 LF method MDE error term 2 (Sine function)
solver4 = error_lax_f2()
Lfmde2=dict()
for c in clist:
    Lfmde2[c]=solver4.solve(x,uinitial,c,a,tend)
    
#Plot: MDE error term 2 - LF method (Sine function)
f, ax = plt.subplots(1,1,figsize=(8,5))
for c in clist:
    ax.plot(x,Lfmde2[c],label='c='+str(c),color=colors[c])
ax.set_xlabel('$x$',size=20)
ax.set_ylabel('$LF-MDE-Second-term$',size=20)
ax.grid()
ax.legend(fontsize=16)
plt.savefig('LF MDE2.png',bbox_inches='tight')

#The Lax-Werendoff algorithm - First error term for the MDE
class error_lax_w1():
    def solve(self,x,uinitial,c,a,tend):
        import numpy as np
        dx = x[1]-x[0]
        dt = c*dx/a
        nx = np.size(x)
        uold = np.zeros(nx)
        uxxx = np.zeros(nx)
        t = np.arange(0,tend,dt)
        uold = np.copy(uinitial)
        for j in t:
            for i in range(2,nx-2):
                uxxx[i] =-(a*dx*dx/6)*((1) -c*c)*(uold[i+2]+2*uold[i-1]-2*uold[i+1] -uold[i-2])/(2*dx*dx*dx)
            uxxx[1] =-(a*dx*dx/6)*((1) -c*c)*(uold[3]+2*uold[0]-2*uold[2] -uold[nx-2])/(2*dx*dx*dx)
            uxxx[0] =-(a*dx*dx/6)*((1) -c*c)*(uold[2]+2*uold[nx-2]-2*uold[1] -uold[nx-3])/(2*dx*dx*dx)
            uxxx[nx-2] =-(a*dx*dx/6)*((1) -c*c)*(uold[1]+2*uold[nx-3]-2*uold[nx-1] -uold[nx-4])/(2*dx*dx*dx)
            uxxx[nx-1] = uxxx[0]
        return uxxx
    
# Solver 5 LW method MDE error term 1 (Sine function)
solver5 = error_lax_w1()
Lwmde1=dict()
for c in clist:
    Lwmde1[c]=solver5.solve(x,uinitial,c,a,tend)
    
#Plot: MDE error term 1 - LW method (Sine function)
f, ax = plt.subplots(1,1,figsize=(8,5))
for c in clist:
    ax.plot(x,Lwmde1[c],label='c='+str(c),color=colors[c]) 
ax.set_xlabel('$x$',size=20)
ax.set_ylabel('$LW-MDE-First-term$',size=20)
ax.grid()
ax.legend(fontsize=16)
plt.savefig('LW MDE1.png',bbox_inches='tight')

#The Lax-Werendoff algorithm - Second error term for the MDE
class error_lax_w2():
    def solve(self,x,uinitial,c,a,tend):
        import numpy as np
        dx = x[1]-x[0]
        dt = c*dx/a
        nx = np.size(x)
        uold = np.zeros(nx)
        uxxxx = np.zeros(nx)
        t = np.arange(0,tend,dt)
        uold = np.copy(uinitial)
        for j in t:
            for i in range(2,nx-2):
                uxxxx[i] =-(a*c)*((1) -(c*c))*(uold[i+2]-4*uold[i-1]-4*uold[i+1]+6*uold[i] +uold[i-2])/(8*dx)
            uxxxx[1] =-(a*c)*((1) -(c*c))*(uold[3]-4*uold[0]-4*uold[2]+ 6*uold[1]+uold[nx-2])/(8*dx)
            uxxxx[0] =-(a*c)*((1) -(c*c))*(uold[2]-4*uold[nx-2]-4*uold[1]+6*uold[0] +uold[nx-3])/(8*dx)
            uxxxx[nx-2] =-(a*c)*((1) -(c*c))*(uold[1]-4*uold[nx-3]-4*uold[nx-1]+6*uold[nx-2] +uold[nx-4])/(8*dx)
            uxxxx[nx-1] = uxxxx[0]
        return uxxxx
    
# Solver 6 LW method MDE error term 2 (Sine function)
solver6 = error_lax_w2()
Lwmde2=dict()
for c in clist:
    Lwmde2[c]=solver6.solve(x,uinitial,c,a,tend)
    
#Plot: MDE error term 2 - LW method (Sine function)
f, ax = plt.subplots(1,1,figsize=(8,5))
for c in clist:
    ax.plot(x,Lwmde2[c],label='c='+str(c),color=colors[c])
ax.set_xlabel('$x$',size=20)
ax.set_ylabel('$LW-MDE-second-term$',size=20)
ax.grid()
ax.legend(fontsize=16)
plt.savefig('LW MDE2.png',bbox_inches='tight')

#PART 4: Comparison of the leading order truncation terms(Analytical and Numerical) for n=1
# For the LF method:Analytical error for uxx(e1)
uinitial = (1. + np.sin(2.*np.pi*n*x))/2
uexact = (1. + np.sin(2.*np.pi*n*zeta))/2

c=0.5
uxx=-2*((np.pi)**2)*n*np.sin(2*np.pi*n*zeta)
e1=(a*dx/2)*((1/c)-c)*uxx

#Plots: Analytical,Numerical and the u-uexact error plots for uxx term LF method
f, ax = plt.subplots(1,1,figsize=(8,5))
ax.plot(x,e1,label='Analytical error uxx',color='black') 
ax.plot(x,Lfmde1[c],label='Numerical error uxx',color='blue')
ax.plot(x,u_LF_sine[c]-uexact,label='error-x plot LF method',color='green')
ax.set_xlabel('$x$',size=20)
ax.set_ylabel('$Error-LF#$',size=20)
ax.grid()
ax.legend(fontsize=16)
plt.savefig('Part 4 errors LF method.png',bbox_inches='tight')

# For the LW method:Analytical error for uxxx(e2)
uxxx=-4*((np.pi)**3)*n*np.cos(2*np.pi*n*zeta)
e2=(-a*(dx*2)/6)*(1-c**2)*uxxx

#Plots: Analytical,Numerical and the u-uexact error plots for uxxx term LW method
f, ax = plt.subplots(1,1,figsize=(8,5))
ax.plot(x,e2,label='Analytical error uxxx',color='black') 
ax.plot(x,Lwmde1[c],label='Numerical error uxxx',color='blue')
ax.plot(x,u_LW_sine[c]-uexact,label='error-x plot LW method',color='green')
ax.set_xlabel('$x$',size=20)
ax.set_ylabel('$Error-LW$',size=20)
ax.grid()
ax.legend(fontsize=16)
plt.savefig('Part 4 errors LW method.png',bbox_inches='tight')

#PART 5: Comparison of the leading order truncation terms(Analytical and Numerical) for n=3
# For the LF method:Analytical error for uxx(e1)
uinitial = (1. + np.sin(2.*np.pi*n*x))/2
uexact = (1. + np.sin(2.*np.pi*n*zeta))/2

# For the LF method:Analytical error for uxx(e1)
c=0.5
uxx=-2*((np.pi)**2)*n*np.sin(2*np.pi*n*zeta)
e1=(a*dx/2)*((1/c)-c)*uxx

#Plots: Analytical,Numerical and the u-uexact error plots for uxx term LF method
f, ax = plt.subplots(1,1,figsize=(8,5))
ax.plot(x,e1,label='Analytical error uxx',color='black') 
ax.plot(x,Lfmde1[c],label='Numerical error uxx',color='blue')
ax.plot(x,u_LF_sine[c]-uexact,label='error-x plot LF method',color='green')
ax.set_xlabel('$x$',size=20)
ax.set_ylabel('$Error-LF#$',size=20)
ax.grid()
ax.legend(fontsize=16)
plt.savefig('Part 4 errors LF method.png',bbox_inches='tight')

# For the LW method:Analytical error for uxxx(e2)
uxxx=-4*((np.pi)**3)*n*np.cos(2*np.pi*n*zeta)
e2=(-a*(dx*2)/6)*(1-c**2)*uxxx

#Plots: Analytical,Numerical and the u-uexact error plots for uxxx term LW method
f, ax = plt.subplots(1,1,figsize=(8,5))
ax.plot(x,e2,label='Analytical error uxxx',color='black') 
ax.plot(x,Lwmde1[c],label='Numerical error uxxx',color='blue')
ax.plot(x,u_LW_sine[c]-uexact,label='error-x plot LW method',color='green')
ax.set_xlabel('$x$',size=20)
ax.set_ylabel('$Error-LW$',size=20)
ax.grid()
ax.legend(fontsize=16)
plt.savefig('Part 4 errors LW method.png',bbox_inches='tight')
