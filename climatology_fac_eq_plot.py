from polplot import pp
import numpy as np
from pyamps import AMPS
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text',usetex=True)
#rc('text.latex', preamble='\usepackage{color}')

d2r = np.pi/180

v = 350
B = 4.
tilts = [-25, 0, 25]
F107 = 80.
EQRES = 10
model = AMPS(v, B, B, 0, F107, minlat = 60) # CHANGE COEFFICIENT FILE IN THIS LINE

psilevels = np.hstack((np.r_[-300:0:EQRES], np.r_[0:300:EQRES]))
faclevels = np.linspace(-.95, .95, 20)/2
mlat, mlt = map(lambda x:np.split(x, 2)[0], model.scalargrid)

withFAC = True

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
    if withFAC:
        axcbar = plt.subplot2grid((91, 3), (90, 1))
    axinfo = plt.subplot2grid((91, 3), (90, 0))
    axinfo.set_axis_off()
    
    for i in range(3): 
        for j in range(3):
            if i == j == 1:
                continue # skip
            By, Bz = np.sin(ca[(i,j)] * d2r) * B, np.cos(ca[(i,j)] * d2r) * B
            axdial.text(np.sin(ca[i, j] * d2r), np.cos(ca[i, j] * d2r), str(ca[i, j]) + r'$^\circ$', rotation = -ca[i, j], ha = 'center', va = 'center')
            axdial.plot([.75*np.sin(ca[i, j] * d2r), .85*np.sin(ca[i, j] * d2r)], [.75*np.cos(ca[i, j] * d2r), .85*np.cos(ca[i, j] * d2r)], color = 'lightgrey')
    
            if (i, j) == (1, 2):
                j = 1
            model.update_model(v, By, Bz, tilt, F107)
        
            ju      = model.get_upward_current(mlat, mlt)
            psi     = model.get_divergence_free_current_function(mlat, mlt)
            j_up, j_down, _, __ = model.get_integrated_upward_current()
            if withFAC:
                axes[i][j].contourf(mlat, mlt, ju, levels = faclevels, cmap = plt.cm.bwr, extend = 'both')
                axes[i][j].write(60, 21, '$\parallel$ {:.1f}'.format(j_up) + ' MA\n$\perp$ {0:.0f} kA'.format(psi.max() - psi.min()), ha = 'right', va = 'top', size = 9, multialignment='right')
                axes[i][j].write(60, 10, '$\Delta+' + '{:.2f}'.format(ju.max()) + '\mu$A/m$^2$\n $\\nabla' + '{:.2f}'.format(ju.min()) + ' \mu$A/m$^2$', ha = 'left', va = 'bottom', size = 9, multialignment='left')
                #axes[i][j].write(60,  2, '{0:.0f} kA'.format(psi.max() - psi.min()), ha = 'left', va = 'top', size = 9, multialignment='left')
                axes[i][j].scatter(mlat[ju.argmax()], mlt[ju.argmax()], marker = '^', s = 40, c = 'black', zorder = 100)    
                axes[i][j].scatter(mlat[ju.argmax()], mlt[ju.argmax()], marker = '^', s = 20, c = 'red', linewidth = .5, zorder = 101)    
                axes[i][j].scatter(mlat[ju.argmin()], mlt[ju.argmin()], marker = 'v', s = 40, c = 'black', zorder = 100)    
                axes[i][j].scatter(mlat[ju.argmin()], mlt[ju.argmin()], marker = 'v', s = 20, c = 'blue', linewidth = .5, zorder = 101)    

            axes[i][j].contour( mlat, mlt, psi, levels = psilevels, colors = 'black', linewidths = .8)
        
    
    
    
    axdial.text(0, 0, u'DIVERGENCE-\nFREE AND\nBIRKELAND\nCURRENT', ha = 'center', va = 'center', size = 14)
    
    
    
    plt.subplots_adjust(hspace = .01, wspace = .01, left = .01, right = .99, bottom = .05, top = .99)
    
    axdial.set_xlim(-1.2, 1.2)
    axdial.set_ylim(-1.2, 1.2)
    
    if withFAC:
        axcbar.contourf(faclevels, [0, 1], np.vstack((faclevels, faclevels)), levels = faclevels, cmap = plt.cm.bwr)
        axcbar.set_yticks([])
        axcbar.set_xlabel('$\mu$A/m$^2$')
    
    
    axinfo.text(axinfo.get_xlim()[0], axinfo.get_ylim()[0], '$|B|$ = %s nT, $v$ = %s km/s,\n F$_{10.7}$ = %s, TILT $= %s^\circ$' % (B, v, F107, tilt), ha = 'left', va = 'top', size = 9)

    plt.savefig('amps_1.2_equivalent_fac_current_' + str(tilt) + '_deg_tilt.png', dpi = 250)
    plt.savefig('amps_1.2_equivalent_fac_current_' + str(tilt) + '_deg_tilt.pdf')
    plt.savefig('amps_1.2_equivalent_fac_current_' + str(tilt) + '_deg_tilt.svg')
    
plt.show()
