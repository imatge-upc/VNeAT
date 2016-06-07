print 'The structures of the program are being initialized, please be pacient.'

from matplotlib.pyplot import show, plot
from numpy import array as nparray, linspace

import database as db
from curve_fit_v2 import GLM
from tools import polynomial, tolist

input_data = db.get_data(x1=0, y1=0, z1=0, x2=1, y2=1, z2=1)

subjects = input_data.subjects
diags = subjects[0].diagnostics
diag = [[] for _ in diags]

# Compute features (independent of voxel, dependent only on subjects)
sex = []
age = []
adcsf = []
for i in range(len(subjects)):
    subject = subjects[i]
    sex.append(subject.sex)
    age.append(subject.age)
    adcsf.append(subject.adcsf)
    diag[subject.diag].append(i)

# Sex is set only linearly (does not make sense to power it, since sex**(2*k) = ones(l),
# and sex**(2*k + 1) = sex for all k)
xdata1 = [sex]

# Polynomials up to 3rd degree of age and adcsf
for p in polynomial(2, [age]):
    xdata1.append(p)

xdata1 = nparray(xdata1, dtype=float).T

# Correction GLM
glm1 = GLM(xdata1, xdata1[:, 0], homogeneous=True)
glm1.orthonormalize()

# Polynomyals up to 3 of extended AD-CSF index axis to compute output
ladcsf = min(adcsf)
radcsf = max(adcsf)
npoints = 100
adcsf_axis = linspace(ladcsf, radcsf, npoints)
adcsf_polys = nparray(tolist(polynomial(3, [adcsf_axis]))).T
adcsf = nparray(adcsf)

output_file = db.open_output_file('/Users/Asier/Documents/TFG/python/output_v6_2.nii')
output_data = output_file.get_data()

while True:
    try:
        entry = raw_input('Write a tuple of voxel coordinates to display its curve (or press Ctrl+D to exit): ')
    except EOFError:
        print
        print 'Thank you for using our service.'
        print
        break
    except Exception as e:
        print '[ERROR] Unexpected error was found when reading input:'
        print e
        print
        continue
    try:
        x, y, z = map(int, eval(entry))
    except (NameError, TypeError, ValueError, EOFError):
        print '[ERROR] Input was not recognized'
        print 'To display the voxel with coordinates (x, y, z), please enter \'x, y, z\''
        print 'e.g., for voxel (57, 49, 82), type \'57, 49, 82\' (without inverted commas) as input'
        print
        continue
    except Exception as e:
        print '[ERROR] Unexpected error was found when reading input:'
        print e
        print
        continue

    print 'Processing request... please wait'

    try:
        vgen = db.get_data(x1=x, x2=x + 1, y1=y, y2=y + 1, z1=z, z2=z + 1)
        vgen = vgen.voxels()
        v = vgen.next()

        params = output_data[:4, x, y, z]
        corrected_data = v.data - GLM.predict(glm1.xdata, params)

        params_l = output_data[4:5, x, y, z]
        lin_curve = GLM.predict(adcsf_polys[:, :1], params_l)

        params_nl = output_data[5:, x, y, z]
        nonlin_curve = GLM.predict(adcsf_polys[:, 1:], params_nl)

        params = output_data[4:, x, y, z]
        curve = GLM.predict(adcsf_polys, params)

        plot(adcsf_axis, lin_curve, 'r', label='Fitted linear curve')
        plot(adcsf_axis, nonlin_curve, 'y', label='Fitted nonlinear curve')
        plot(adcsf_axis, curve, 'k', label='Fitted total curve')

        color = ['go', 'bo', 'mo', 'ko']
        for i in range(len(diag)):
            l = diag[i]
            plot(adcsf[l], corrected_data[l], color[i], label=diags[i])
        # legend()
        show()
        print
    except Exception as e:
        print '[ERROR] Unexpected error occured while computing and showing the results:'
        print e
        print
        continue
