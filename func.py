import numpy as np
y0=10
y1=12
d0=0.3
d1=0.6
d2=2.0
d3=1.3
def fun_I(t):	
	y=-1/490000*pow(t,2)+1400/490000*t
	return(y)
def fun_U(t):
	if t>=0 and t<300:	
		y=0.125
	if t>=300 and t<500:
		y=0.25
	if t>=500 and t<900:
		y=0.5
	if t>=900 and t<1100:
		y=0.25
	if t>=1100:
		y=0.125
	return(y)
def NARMA(t,m):
	u=np.zeros(m)
	y=np.zeros(m)	
	I=fun_I(t)
	y[0]=y0
	y[1]=y1
	for i in range (m):
		u[i]=fun_U(t[i])
	for i in range (m):
		if t[i]<=0:
			y[i]=0
		y[i]=(1-d1)*y[i-1]+(1-d3)*y[i-1]*u[i-1]/u[i-2]+(d3-1)*(1+d1)*y[i-2]*u[i-1]/u[i-2]+d0*d2*u[i-1]*I[i-2]-d0*u[i-2]*y[i-1]+d0*(1+d1)*u[i-1]*y[i-2]
	return y
