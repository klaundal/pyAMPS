from polplot import pp
import numpy as np
from pyamps import AMPS
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text',usetex=True)
rc('text.latex', preamble=r'\usepackage{color}')

d2r = np.pi/180

v = 350
B = 4.
tilts = [-25, 0, 25]
F107 = 80.

modelv = AMPS(v, B, B, 0, F107, minlat = 60)
models = AMPS(v, B, B, 0, F107, minlat = 60)

clevels = np.linspace(0, 250, 11)
# remove the southern hemisphere:
modelv.vectorgrid = list(map(lambda x:np.split(x, 2)[0], modelv.vectorgrid))
modelv.scalargrid = list(map(lambda x:np.split(x, 2)[0], modelv.scalargrid))
modelv.calculate_matrices()
mlatv, mltv = modelv.vectorgrid
mlats, mlts = modelv.scalargrid

# for models, replace vectorgrdid by scalargrid
models.vectorgrid = modelv.scalargrid 
models.calculate_matrices()

# dictionary of where to put the different clock angle bins
ca = {(0, 1): 0, (0, 0):-45, (1, 0):-90,(2,0):-135,(2,1):180,(0,2):45,(1,2):90,(2,2):135}

for tilt in tilts:#tilt = tilts[0]

    fig = plt.figure(figsize = (9, 9))
    axes = [[pp(plt.subplot2grid((91, 3), (j*30, i), rowspan = 30), 
                          linestyle = ':', color = 'grey', minlat = 60, linewidth = .9)
             for i in range(3) if (i, j) != (1, 1)] 
             for j in range(3)]
    axdial = plt.subplot2grid((91, 3), (30, 1), rowspan = 30)
    axdial.set_aspect('equal')
    axdial.set_axis_off()
    axcbar = plt.subplot2grid((91, 3), (90, 1))
    axinfo = plt.subplot2grid((91, 3), (90, 0))
    axtemplate = plt.subplot2grid((91, 3), (90, 2))
    axinfo.set_axis_off()
    axtemplate.set_axis_off()
    axtemplate.set_xlim(-1.1, 1.1)
    axtemplate.set_ylim(-3, 3)
    
    for i in range(3): 
        for j in range(3):
            if i == j == 1:
                continue # skip
            By, Bz = np.sin(ca[(i,j)] * d2r) * B, np.cos(ca[(i,j)] * d2r) * B
            axdial.text(np.sin(ca[i, j] * d2r), np.cos(ca[i, j] * d2r), str(ca[i, j]) + r'$^\circ$', rotation = -ca[i, j], ha = 'center', va = 'center')
            axdial.plot([.75*np.sin(ca[i, j] * d2r), .85*np.sin(ca[i, j] * d2r)], [.75*np.cos(ca[i, j] * d2r), .85*np.cos(ca[i, j] * d2r)], color = 'lightgrey')
    
            if (i, j) == (1, 2):
                j = 1
            modelv.update_model(v, By, Bz, tilt, F107)
            models.update_model(v, By, Bz, tilt, F107)
        
            jev, jnv = modelv.get_total_current()
            jes, jns = models.get_total_current()
            jtot = np.sqrt(jes**2 + jns**2)
            axes[i][j].contourf(mlats, mlts, jtot, levels = clevels, extend = 'both', cmap = plt.cm.viridis)
            axes[i][j].plotpins(mlatv, mltv, jnv, jev, SCALE = 100, markersize = 1, markercolor = 'black', color = 'black', linewidth = .5, unit = None)
            axes[i][j].write(60, 10, '*' + str(int(np.round(jtot.max()))) + ' mA/m', ha = 'left', va = 'bottom', size = 9)
            axes[i][j].scatter(mlats[jtot.argmax()], mlts[jtot.argmax()], marker = (6, 2, 0), s = 30, c = 'black', zorder = 100)    
            axes[i][j].scatter(mlats[jtot.argmax()], mlts[jtot.argmax()], marker = (6, 2, 0), s = 30, c = 'gold', linewidth = .5, zorder = 101)

    
    
    axdial.text(0, 0, u'TOTAL\nCURRENT', ha = 'center', va = 'center', size = 14)
    
    
    
    plt.subplots_adjust(hspace = .01, wspace = .01, left = .01, right = .99, bottom = .05, top = .99)
    
    axdial.set_xlim(-1.1, 1.1)
    axdial.set_ylim(-1.1, 1.1)

    # add template vector
    #axdial.plot([-0.1, 0.1], [0.35, 0.35], color = 'black', linestyle = '-', linewidth = .9)
    #axdial.scatter(-0.1, 0.35, marker = 'o', s = 4, c = 'black')
    #axdial.text(0, 0.36, '200 mA/m', ha = 'center', va = 'bottom', size = 8)
    axtemplate.plot([.8, 1], [0, 0], color = 'black', linestyle = '-', linewidth = .9)
    axtemplate.scatter(1, 0, marker = 'o', s = 4, c = 'black')
    axtemplate.text(.75, 0, '200 mA/m', ha = 'right', va = 'center', size = 9)

    
    axcbar.contourf(clevels, [0, 1], np.vstack((clevels, clevels)), levels = clevels, cmap = plt.cm.viridis)
    axcbar.set_yticks([])
    axcbar.set_xlabel('mA/m')
    
    
    axinfo.text(axinfo.get_xlim()[0], axinfo.get_ylim()[0], '$|B|$ = %s nT, $v$ = %s km/s,\n F$_{10.7}$ = %s, TILT $= %s^\circ$' % (B, v, F107, tilt), ha = 'left', va = 'top', size = 9)
    plt.savefig('amps_1.2_total_current_' + str(tilt) + '_deg_tilt.png', dpi = 250)
    plt.savefig('amps_1.2_total_current_' + str(tilt) + '_deg_tilt.pdf')

plt.show()
