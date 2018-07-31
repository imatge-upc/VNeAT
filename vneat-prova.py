from nilearn import datasets, plotting
from nibabel.freesurfer import io, mghformat
import matplotlib.pyplot as plt

plt.switch_backend('TkAgg')



filename = '/projects/neuro/ADNI-Screening-1.5T-Registered/4/surf/lh.inflated'
a,b = io.read_geometry(filename)

# fsaverage = datasets.fetch_surf_fsaverage5()
plotting.plot_surf([a,b])
plotting.show()
