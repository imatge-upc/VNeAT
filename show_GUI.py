import matplotlib.pyplot as plt
import numpy as np


class FSLView(object):
    def __init__(self, template, template_cmap='gray'):
        self._template = (template - np.min(template)).astype(float)/(np.max(template) - np.min(template))
        self._template_cmap = template_cmap
        self._images = []
        self._processors = []

    def add_image(self, image, colormap='hot'):
        """Remember that you can mask the image before adding it to the plot as follows:
            # For a case in which we would like to make transparent the voxels of the matrix that contain a value of 0,
            # assuming that original_image is a numpy (np) array:
            masked_image = np.ma.masked_where(original_image == 0, original_image))

            # To make transparent any other voxels, change 'original_image == 0' in the previous line with the condition
            # that the voxels to be masked must fulfill
        """
        if image.shape != self._template.shape:
            raise ValueError("The shape of the image " + str(image.shape) + " must match that of the template " + str(template.shape))

        image = (image - np.min(image)).astype(float)/(np.max(image) - np.min(image))

        self._images.append((image, colormap))
        return self

    def add_curve_processor(self, processor, prediction_parameters, label=None):
        if prediction_parameters.shape[1:] != self._template.shape:
            raise ValueError("The shape of the prediction parameters " + str(prediction_parameters.shape[1:]) + " must match that of the template " + str(template.shape))
        if label is None:
            label = 'Curve ' + str(len(self._processors)+1)
        self._processors.append((processor, prediction_parameters, label))
        return self

    def set_corrected_data_processor(self, processor, correction_parameters, axis):
        if correction_parameters.shape[1:] != self._template.shape:
            raise ValueError("The shape of the correction parameters " + str(correction_parameters.shape[1:]) + " must match that of the template " + str(template.shape))

        self._correction_processor = (processor, correction_parameters, axis)

    def __compute_new_voxel_coords__(self, event):
        if event.inaxes is self._ax[0,0]:
            new_voxel = event.xdata, self._current_voxel[1], (self._template.shape[2]-1)-event.ydata
        elif event.inaxes is self._ax[0,1]:
            new_voxel = self._current_voxel[0], (self._template.shape[1]-1)-event.xdata, (self._template.shape[2]-1)-event.ydata
        elif event.inaxes is self._ax[1,0]:
            new_voxel = event.xdata, (self._template.shape[1]-1)-event.ydata, self._current_voxel[2]
        else:
            return None

        return tuple(int(0.5 + new_voxel[i]) for i in xrange(len(new_voxel)))

    def __button_press_event__(self, event):
        if event.button != 1:
            return

        self.__update_views__(self.__compute_new_voxel_coords__(event))


        # if event.inaxes is self._ax[0,0]:
        #    new_voxel = event.xdata, event.ydata

#    def __button_release_event__(self, event):
#        print 'Button released!', event

    def __motion_notify_event__(self, event):
        if event.button != 1:
            return
        
        self.__update_views__(self.__compute_new_voxel_coords__(event))

#    def __axes_leave_event__(self, event):
#        print 'Axis left!', event

    @staticmethod
    def __axial_cut__(image, voxel):
        return np.fliplr(image[:, :, voxel[2]]).T

    @staticmethod
    def __sagittal_cut__(image, voxel):
        return np.fliplr(np.fliplr(image[voxel[0], :, :]).T)

    @staticmethod
    def __coronal_cut__(image, voxel):
        return np.fliplr(image[:, voxel[1], :]).T

    def __update_views__(self, new_voxel):
        if new_voxel is None or new_voxel == self._current_voxel:
            return

        axes = [(self._ax[0,1], self.__sagittal_cut__), (self._ax[0,0], self.__coronal_cut__), (self._ax[1,0], self.__axial_cut__)]
        for i in xrange(len(axes)):
            if new_voxel[i] == self._current_voxel[i]:
                continue
            ax, cut = axes[i]
            for img, cmap in [(self._template, self._template_cmap)] + self._images:
                ax.imshow(cut(img, new_voxel), cmap=cmap)

        self._ax[1,1].clear()

        x, y, z = new_voxel

        colors = ['r', 'c', 'g', 'm', 'y', 'k']

        try:
            correction_processor, correction_parameters, axis = self._correction_processor
            cdata = correction_processor.corrected_values(correction_parameters,
                x1=x, x2=x + 1, y1=y, y2=y + 1, z1=z, z2=z + 1)
            self._ax[1,1].plot(axis, cdata[:, 0, 0, 0], 'bo')
        except AttributeError:
            pass

        for processor, prediction_parameters, label in self._processors:
            axis, curve = processor.curve(prediction_parameters,
                x1=x, x2=x + 1, y1=y, y2=y + 1, z1=z, z2=z + 1, tpoints=50
            )

            self._ax[1,1].plot(axis, curve[:, 0, 0, 0],
                            label=label, color=colors[0], marker='d')

            colors.append(colors[0])
            del colors[0]

        if len(self._processors) > 0 or hasattr(self, '_correction_processor'):
            self._ax[1,1].legend()

        self._figure.canvas.draw()

        self._current_voxel = new_voxel

    def show(self):
        self._figure = plt.figure()

        outer_padding = (0.04, 0.04, 0.04, 0.04) # left, right, bottom, up
        inner_padding = (0.02, 0.02) # horizontal, vertical
        
        total_width = self._template.shape[0] + self._template.shape[1]
        total_height = self._template.shape[2] + self._template.shape[1]

        effective_width = 1. - outer_padding[0] - outer_padding[1] - inner_padding[0]
        effective_height = 1. - outer_padding[2] - outer_padding[3] - inner_padding[1]


        width1 = self._template.shape[0]*effective_width/total_width
        width2 = effective_width - width1

        height1 = self._template.shape[2]*effective_height/total_height
        height2 = effective_height - height1

        self._ax = np.array([
            self._figure.add_axes([outer_padding[0], outer_padding[2] + height2 + inner_padding[1], width1, height1]),
            self._figure.add_axes([outer_padding[0] + width1 + inner_padding[0], outer_padding[2] + height2 + inner_padding[1], width2, height1]),
            self._figure.add_axes([outer_padding[0], outer_padding[2], width1, height2]),
            self._figure.add_axes([outer_padding[0] + width1 + inner_padding[0], outer_padding[2], width2, height2])
        ]).reshape((2, 2))

        for ax in self._ax[0,0], self._ax[0,1], self._ax[1,0]:
            ax.set_xticks([])
            ax.set_yticks([])

#       self._figure, self._ax = plt.subplots(2, 2, gridspec_kw = {'height_ratios':[1, 1]})

        self._current_voxel = 0, 0, 0

        self._figure.canvas.mpl_connect('button_press_event', self.__button_press_event__)
#        self._figure.canvas.mpl_connect('button_release_event', self.__button_release_event__)
        self._figure.canvas.mpl_connect('motion_notify_event', self.__motion_notify_event__)
#        self._figure.canvas.mpl_connect('axes_leave_event', self.__axes_leave_event__)

        new_voxel = map(lambda x: x/2, self._template.shape)

        self.__update_views__(new_voxel)
#        self._figure.tight_layout()

        plt.show()



if __name__ == '__main__':
    from Utils.DataLoader import getSubjects, getMNITemplate
    from Utils.Subject import Subject
    from Processors.GLMProcessing import GLMProcessor
    import nibabel as nib
    import matplotlib.cm as cm
    from os.path import join
    template = getMNITemplate()

    fn = join('/', 'Users', 'asier', 'Documents', 'git', 'imatge-upc', 'neuroimatge', 'results', 'GLM', 'glm_all_')
    zscores = nib.load(fn + 'zscores_0.999.nii.gz').get_data()
    masked_zscores = np.ma.masked_where(zscores == 0, zscores)
    pparams = nib.load(fn + 'intercept_pparams.nii.gz').get_data()
    cparams = nib.load(fn + 'intercept_cparams.nii.gz').get_data()

    with open(fn + 'intercept_userdefparams.txt', 'rb') as f:
        user_def_params = eval(f.read())

    subjects = getSubjects()
    glmp = GLMProcessor(subjects, predictors=[Subject.ADCSFIndex], correctors=[Subject.Age, Subject.Sex], user_defined_parameters=user_def_params)

    fslview = FSLView(template, 'gray')
    fslview.add_image(masked_zscores, 'hot')
    fslview.add_curve_processor(glmp, pparams, 'GLM')
    fslview.set_corrected_data_processor(glmp, cparams, glmp.predictors[:, 0])

    fslview.show()

