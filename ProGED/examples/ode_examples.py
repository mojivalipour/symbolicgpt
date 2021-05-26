#%% dataset
import numpy as np
# import matplotlib.pyplot as plt
# from scipy.integrate import solve_ivp #,  odeint
# from scipy.interpolate import interp1d

# # %%
# # urejeni exampli:

# # diff eq no.1 ( dot{x} = a*x + c )

# def ode_plot(ts, Xs, dx):
#     Xode = solve_ivp(dx, (ts[0],ts[-1]), Xs[:1], t_eval=ts)
#     plt.plot(ts,Xs,"r-")
#     plt.plot(ts, Xode.y[0],'k--')
#     error = sum((Xs-Xode.y[0])**2)
#     print(f"Napaka ode: {error}")
#     return error
# # print(ode_plot(ts, Xs_c, Ys_c, dx_c))

# def example_c_data(
#     n = 1000,
#     t_start = 0.45,
#     t_end = 0.87,
#     C = 3.75,
#     cy = 10.34,
#     a = 0.4
#     ):
#     ts = np.linspace(t_start, t_end, n)
#     Xs_c = C*np.exp(a*ts) - cy/a
#     Ys_c = np.ones(ts.shape[0])*cy
#     return ts, Xs_c, Ys_c, cy, a

# def example1c(n=1000, t1=0.45, t2=0.87, C=3.75, cy=10.34, a=0.4):
#     ts, Xs_c, _, cy, a = example_c_data(n,t1,t2,C,cy,a)
#     def dx_c(t,x):
#         return a*x + cy
#     return ode_plot(ts, Xs_c, dx_c)
# # print(example1c())

# def example2c(n=1000, t1=0.45, t2=0.87, C=3.75, cy=10.34, a=0.4):
#     ts, Xs_c, Ys_c, _, a = example_c_data(n,t1,t2,C,cy,a)
#     def y_index(t, Ys, ts):
#         """vrne vrednost v stolpcu y, ki pripada casu t"""
#         find_index = ts==t # = [False, False, True, False, ...]
#         index_t = find_index.argmax() # vrne mesto t-ja v stolpcu T
#         return Ys[index_t] # vrne (index_t)-to vrednost v stolpcu Y
#     def dx_c(t, x):
#         return a*x + y_index(t, Ys_c, ts)
#     return ode_plot(ts, Xs_c, dx_c)
# # print(example2c())

# def example3c(n=1000, t1=0.45, t2=0.87, C=3.75, cy=10.34, a=0.4):
#     ts, Xs_c, Ys_c, _, a = example_c_data(n,t1,t2,C,cy,a)
#     y = interp1d(ts, Ys_c, kind='cubic')
#     def dx_c(t, x):
#         return a*x + y(t)
#     return ode_plot(ts, Xs_c, dx_c)
# # print(example3c())

# # print(example1c())
# # print(example2c())
# # print(example3c())



# diff eq no.2 ( dot{x} = a*x + e^(a*t) )

def example_tB_data(
    n = 1000,
    t_start = 0.45,
    t_end = 0.87,
    B = -2.56, # B = 0,
    a = 0.4
    ):
    ts = np.linspace(t_start, t_end, n)
    Xs = (ts+B)*np.exp(a*ts)
    Ys = np.exp(a*ts)
    return ts, Xs, Ys, B, a

# def example4tB(n=1000, t1=0.45, t2=0.87, B=-2.56, a=0.4):
#     ts, Xs, Ys, _, a = example_tB_data(n,t1,t2,B,a)
#     def y_index(t, Ys, ts):
#         """vrne vrednost v stolpcu y, ki pripada casu t"""
#         find_index = ts==t # = [False, False, True, False, ...]
#         index_t = find_index.argmax() # vrne mesto t-ja v stolpcu T
#         return Ys[index_t] # vrne (index_t)-to vrednost v stolpcu Y
#     def dx_index(t, x):
#         return a*x + y_index(t, Ys, ts)
#     return ode_plot(ts, Xs, dx_index)
# # print(example4tB())

# def example5tB(n=1000, t1=0.45, t2=0.87, B=-2.56, a=0.4):
#     ts, Xs, Ys, _, a = example_tB_data(n,t1,t2,B,a)
#     y = interp1d(ts, Ys, kind='cubic')
#     def dx(t, x):
#         return a*x + y(t)
#     return ode_plot(ts, Xs, dx)
# # print(example5tB())


# # diff eq no.3

# def example_ab_data(
#     n = 1000,
#     t_start = 0.45,
#     t_end = 0.87,
#     C = 4.21,
#     b = 4, # b=-15.76,
#     a = 0.4
#     ):
#     ts = np.linspace(t_start, t_end, n)
#     Xs = (1/(b-a))*np.exp(b*ts) + C*np.exp(a*ts)
#     Ys = np.exp(b*ts)
#     return ts, Xs, Ys, a

# def example6ab(n=1000, t1=0.45, t2=0.87, C=4.21, b=4, a=0.4):
#     ts, Xs, Ys, a = example_ab_data(n,t1,t2,C,b,a)
#     def y_index(t, Ys, ts):
#         """vrne vrednost v stolpcu y, ki pripada casu t"""
#         find_index = ts==t # = [False, False, True, False, ...]
#         index_t = find_index.argmax() # vrne mesto t-ja v stolpcu T
#         return Ys[index_t] # vrne (index_t)-to vrednost v stolpcu Y
#     def dx_index(t, x):
#         return a*x + y_index(t, Ys, ts)
#     return ode_plot(ts, Xs, dx_index)
# # print(example6ab())  # pomankljivost indexa
# # print(example6ab(b=-15.76))  # delujoc
# # print(example6ab(b=4))  # to je default nastavitev b-ja.

# def example7ab(n=1000, t1=0.45, t2=0.87, C=4.21, b=4, a=0.4):
#     ts, Xs, Ys, a = example_ab_data(n,t1,t2,C,b,a)
#     y = interp1d(ts, Ys, kind='cubic')
#     def dx(t, x):
#         return a*x + y(t)
#     return ode_plot(ts, Xs, dx)
# # print(example7ab())
# # print(example7ab(b=4)) # = original (interpol premaga index)
# # print(example7ab(b=-30, a=-40))  # primer ko "ne deluje"
# # # dodatni primeri (isti) za arhiv:
# # # # print(example7ab(n=1000, t1=0.45, t2=0.87, C=-1*24.21, b=-34.56, a=-40.4))  
# # # # print(example7ab(n=1000, t1=0.45, t2=0.87, C=-1.21, b=-23.56, a=-34.4))

# print(example6ab())
# # print(example7ab(t1=0.95, t2=0.87))

