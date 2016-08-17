import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D as mplline


class GUIVisualizer(object):

    def __init__(self, template, num_points=50, template_cmap='gray'):
        self._template = (template - np.min(template)).astype(float) / (np.max(template) - np.min(template))
        self._num_points = num_points
        self._template_cmap = template_cmap
        self._images = []
        self._processors = []
        self._correction_processor = None
        self._figure = None
        self._ax = None
        self._rgba_image = None
        self._current_voxel = None

    def add_image(self, image, colormap='hot'):
        """
        Remember that you can mask the image before adding it to the plot as follows:
        For a case in which we would like to make transparent the voxels of the matrix that contain a value of 0,
        assuming that original_image is a numpy (np) array:
        masked_image = np.ma.masked_where(original_image == 0, original_image))

        To make transparent any other voxels, change 'original_image == 0' in the previous line with the condition
        that the voxels to be masked must fulfill
        """
        if image.shape != self._template.shape:
            raise ValueError("The shape of the image " + str(image.shape) + " must match that of the template " + str(
                self._template.shape))

        image = (image - np.min(image)).astype(float) / (np.max(image) - np.min(image))

        self._images.append((image, colormap))
        return self

    def add_curve_processor(self, processor, prediction_parameters, label=None):
        if prediction_parameters.shape[1:] != self._template.shape:
            raise ValueError("The shape of the prediction parameters " + str(
                prediction_parameters.shape[1:]) + " must match that of the template " + str(self._template.shape))
        if label is None:
            label = 'Curve ' + str(len(self._processors) + 1)
        self._processors.append((processor, prediction_parameters, label))
        return self

    def set_corrected_data_processor(self, processor, correction_parameters):
        if correction_parameters.shape[1:] != self._template.shape:
            raise ValueError("The shape of the correction parameters " + str(
                correction_parameters.shape[1:]) + " must match that of the template " + str(self._template.shape))

        self._correction_processor = (processor, correction_parameters)

    def __compute_new_voxel_coords__(self, event):
        if event.inaxes is self._ax[0, 0]:
            new_voxel = event.xdata, self._current_voxel[1], (self._template.shape[2] - 1) - event.ydata
        elif event.inaxes is self._ax[0, 1]:
            new_voxel = self._current_voxel[0], (self._template.shape[1] - 1) - event.xdata, (
                self._template.shape[2] - 1) - event.ydata
        elif event.inaxes is self._ax[1, 0]:
            new_voxel = event.xdata, (self._template.shape[1] - 1) - event.ydata, self._current_voxel[2]
        else:
            return None

        return tuple(int(0.5 + new_voxel[i]) for i in xrange(len(new_voxel)))

    def __compute_xydata__(self, voxel):
        xydata00 = voxel[0], (self._template.shape[2] - 1) - voxel[2]
        shape00 = self._template.shape[0], self._template.shape[2]

        xydata01 = (self._template.shape[1] - 1) - voxel[1], (self._template.shape[2] - 1) - voxel[2]
        shape01 = self._template.shape[1], self._template.shape[2]

        xydata10 = voxel[0], (self._template.shape[1] - 1) - voxel[1]
        shape10 = self._template.shape[0], self._template.shape[1]

        return (xydata00, shape00), (xydata01, shape01), (xydata10, shape10)

    def __button_press_event__(self, event):
        if event.button != 1:
            return

        self.__update_views__(self.__compute_new_voxel_coords__(event))

    def __motion_notify_event__(self, event):
        if event.button != 1:
            return

        self.__update_views__(self.__compute_new_voxel_coords__(event))

    def __axial_cut__(self, voxel):
        image = self._rgba_image
        channels = image.shape[3]
        cut = np.zeros((image.shape[1], image.shape[0], channels))
        for i in xrange(channels):
            cut[:, :, i] = np.fliplr(image[:, :, voxel[2], i]).T
        return cut

    def __sagittal_cut__(self, voxel):
        image = self._rgba_image
        channels = image.shape[3]
        cut = np.zeros((image.shape[2], image.shape[1], channels))
        for i in xrange(channels):
            cut[:, :, i] = np.fliplr(np.fliplr(image[voxel[0], :, :, i]).T)
        return cut

    def __coronal_cut__(self, voxel):
        image = self._rgba_image
        channels = image.shape[3]
        cut = np.zeros((image.shape[2], image.shape[0], channels))
        for i in xrange(channels):
            cut[:, :, i] = np.fliplr(image[:, voxel[1], :, i]).T
        return cut

    def __update_views__(self, new_voxel):
        if new_voxel is None or new_voxel == self._current_voxel:
            return

        axes = [(self._ax[0, 0], self.__coronal_cut__), (self._ax[0, 1], self.__sagittal_cut__),
                (self._ax[1, 0], self.__axial_cut__)]
        xydata = self.__compute_xydata__(new_voxel)
        for i in xrange(len(axes)):
            ax, cut = axes[i]
            ax.clear()
            ax.imshow(cut(new_voxel), interpolation='nearest')

            (xdata, ydata), (width, height) = xydata[i]
            ax.add_line(mplline(xdata=[0, width], ydata=[ydata, ydata], linewidth=1, color='green'))
            ax.add_line(mplline(xdata=[xdata, xdata], ydata=[0, height], linewidth=1, color='green'))

        self._ax[1, 1].clear()

        x, y, z = new_voxel

        colors = ['r', 'c', 'g', 'm', 'y', 'k']

        try:
            correction_processor, correction_parameters = self._correction_processor
            cdata = correction_processor.corrected_values(correction_parameters,
                                                          x1=x, x2=x + 1, y1=y, y2=y + 1, z1=z, z2=z + 1)
            self._ax[1, 1].plot(correction_processor.predictors[:, 0], cdata[:, 0, 0, 0], 'bo')
        except AttributeError:
            pass

        for processor, prediction_parameters, label in self._processors:
            axis, curve = processor.curve(prediction_parameters,
                                          x1=x, x2=x + 1, y1=y, y2=y + 1, z1=z, z2=z + 1, tpoints=self._num_points
                                          )

            self._ax[1, 1].plot(axis, curve[:, 0, 0, 0],
                                label=label, color=colors[0], marker='d')

            colors.append(colors[0])
            del colors[0]

        if len(self._processors) > 0 or hasattr(self, '_correction_processor'):
            self._ax[1, 1].legend()

        self._figure.canvas.draw()

        self._current_voxel = new_voxel

    def show(self):
        self._figure = plt.figure()

        outer_padding = (0.04, 0.04, 0.04, 0.04)  # left, right, bottom, up
        inner_padding = (0.02, 0.02)  # horizontal, vertical

        total_width = self._template.shape[0] + self._template.shape[1]
        total_height = self._template.shape[2] + self._template.shape[1]

        effective_width = 1. - outer_padding[0] - outer_padding[1] - inner_padding[0]
        effective_height = 1. - outer_padding[2] - outer_padding[3] - inner_padding[1]

        width1 = self._template.shape[0] * effective_width / total_width
        width2 = effective_width - width1

        height1 = self._template.shape[2] * effective_height / total_height
        height2 = effective_height - height1

        self._ax = np.array([
            self._figure.add_axes([outer_padding[0], outer_padding[2] + height2 + inner_padding[1], width1, height1]),
            self._figure.add_axes(
                [outer_padding[0] + width1 + inner_padding[0], outer_padding[2] + height2 + inner_padding[1], width2,
                 height1]),
            self._figure.add_axes([outer_padding[0], outer_padding[2], width1, height2]),
            self._figure.add_axes([outer_padding[0] + width1 + inner_padding[0], outer_padding[2], width2, height2])
        ]).reshape((2, 2))

        for ax in self._ax[0, 0], self._ax[0, 1], self._ax[1, 0]:
            ax.set_xticks([])
            ax.set_yticks([])

        cmap = cm.get_cmap(self._template_cmap)
        self._rgba_image = cmap(self._template)

        for img, cmap in self._images:
            cmap = cm.get_cmap(cmap)
            img = cmap(img)

            Cfg, Afg = img[:, :, :, :3], img[:, :, :, 3]
            Cbg, Abg = self._rgba_image[:, :, :, :3], self._rgba_image[:, :, :, 3]

            Ar = Afg + Abg * (1 - Afg)
            Cr = np.zeros(Cbg.shape)
            for c in xrange(Cr.shape[3]):
                Cr[:, :, :, c] = Cfg[:, :, :, c] * Afg + Cbg[:, :, :, c] * Abg * (1 - Afg) / Ar

            self._rgba_image[:, :, :, :3] = Cr
            self._rgba_image[:, :, :, 3] = Ar

        self._current_voxel = 0, 0, 0

        self._figure.canvas.mpl_connect('button_press_event', self.__button_press_event__)
        self._figure.canvas.mpl_connect('motion_notify_event', self.__motion_notify_event__)

        new_voxel = map(lambda x: x / 2, self._template.shape)

        self.__update_views__(new_voxel)

        plt.show()
