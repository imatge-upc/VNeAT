import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D as mplline
from nilearn import plotting


class GUIVisualizer(object):
    def __init__(self, template, affine, num_points=100, template_cmap='gray'):
        self._template = (template - np.min(template)).astype(float) / (np.max(template) - np.min(template))
        self._template_shape = template.shape
        self._affine = affine
        self._num_points = num_points
        self._template_cmap = template_cmap
        self._images = []
        self._processors = []
        self._mask = []
        self._atlas = []
        self._atlas_dict = []
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
        if image.shape[:3] != self._template_shape[:3]:
            raise ValueError("The shape of the image " + str(image.shape) + " must match that of the template " + str(
                self._template_shape))

        image = (image - np.min(image)).astype(float) / (np.max(image) - np.min(image))

        self._images.append((image, colormap))
        return self

    def add_mask(self, mask):
        """
        Remember that you can mask the image before adding it to the plot as follows:
        For a case in which we would like to make transparent the voxels of the matrix that contain a value of 0,
        assuming that original_image is a numpy (np) array:
        masked_image = np.ma.masked_where(original_image == 0, original_image))

        To make transparent any other voxels, change 'original_image == 0' in the previous line with the condition
        that the voxels to be masked must fulfill
        """

        self._mask.append(mask)
        return self

    def add_atlas(self, atlas, atlas_dict):
        self._atlas.append(atlas)
        self._atlas_dict.append(atlas_dict)

    def add_curve_processor(self, processor, prediction_parameters, correction_parameters, label=None):
        if prediction_parameters.shape[1:] != self._template_shape:
            raise ValueError("The shape of the prediction parameters " + str(
                prediction_parameters.shape[1:]) + " must match that of the template " + str(self._template_shape))
        if correction_parameters.shape[1:] != self._template_shape:
            raise ValueError("The shape of the correction parameters {} must match that of the template {}".format(
                correction_parameters.shape[1:],
                self._template_shape
            ))
        if label is None:
            label = 'Curve {}'.format(len(self._processors) + 1)
        self._processors.append((processor, prediction_parameters, correction_parameters, label))
        return self

    def set_corrected_data_processor(self, processor, correction_parameters):
        if correction_parameters.shape[1:] != self._template_shape:
            raise ValueError("The shape of the correction parameters " + str(
                correction_parameters.shape[1:]) + " must match that of the template " + str(self._template_shape))

        self._correction_processor = (processor, correction_parameters)

    def __compute_new_voxel_coords__(self, event):
        if event.inaxes is self._ax[0, 0]:
            new_voxel = event.xdata, self._current_voxel[1], (self._template_shape[2] - 1) - event.ydata
        elif event.inaxes is self._ax[0, 1]:
            new_voxel = self._current_voxel[0], (self._template_shape[1] - 1) - event.xdata, (
                self._template_shape[2] - 1) - event.ydata
        elif event.inaxes is self._ax[1, 0]:
            new_voxel = event.xdata, (self._template_shape[1] - 1) - event.ydata, self._current_voxel[2]
        else:
            return None

        return tuple(int(0.5 + new_voxel[i]) for i in range(len(new_voxel)))

    def __compute_xydata__(self, voxel):
        xydata00 = voxel[0], (self._template_shape[2] - 1) - voxel[2]
        shape00 = self._template_shape[0], self._template_shape[2]

        xydata01 = (self._template_shape[1] - 1) - voxel[1], (self._template_shape[2] - 1) - voxel[2]
        shape01 = self._template_shape[1], self._template_shape[2]

        xydata10 = voxel[0], (self._template_shape[1] - 1) - voxel[1]
        shape10 = self._template_shape[0], self._template_shape[1]

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
        for i in range(channels):
            cut[:, :, i] = np.fliplr(image[:, :, voxel[2], i]).T
        return cut

    def __sagittal_cut__(self, voxel):
        image = self._rgba_image
        channels = image.shape[3]
        cut = np.zeros((image.shape[2], image.shape[1], channels))
        for i in range(channels):
            cut[:, :, i] = np.fliplr(np.fliplr(image[voxel[0], :, :, i]).T)
        return cut

    def __coronal_cut__(self, voxel):
        image = self._rgba_image
        channels = image.shape[3]
        cut = np.zeros((image.shape[2], image.shape[0], channels))
        for i in range(channels):
            cut[:, :, i] = np.fliplr(image[:, voxel[1], :, i]).T
        return cut

    def __update_views__(self, new_voxel):
        if new_voxel is None or new_voxel == self._current_voxel:
            return

        axes = [(self._ax[0, 0], self.__coronal_cut__), (self._ax[0, 1], self.__sagittal_cut__),
                (self._ax[1, 0], self.__axial_cut__)]
        xydata = self.__compute_xydata__(new_voxel)
        for i in range(len(axes)):
            ax, cut = axes[i]
            ax.clear()
            ax.imshow(cut(new_voxel), interpolation='nearest')

            (xdata, ydata), (width, height) = xydata[i]
            ax.add_line(mplline(xdata=[0, width], ydata=[ydata, ydata], linewidth=1, color='green'))
            ax.add_line(mplline(xdata=[xdata, xdata], ydata=[0, height], linewidth=1, color='green'))
            ax.tick_params(direction='in')
            for item in ax.get_xticklabels():
                item.set_fontsize(8)
            for item in ax.get_yticklabels():
                item.set_fontsize(8)

        self._ax[1, 1].clear()

        x, y, z = new_voxel

        if self._images[-1][1] != 'rgb':
            cmap = cm.get_cmap(self._images[-1][1])
            n_colors_list = [0] + [i*int(cmap.N/(len(self._processors)-1)) - 1 for i in range(1,len(self._processors))]
        else:
            cmap = lambda x: [(1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0)][x]
            n_colors_list = list(range(len(self._processors)))

        colors = [cmap(i) for i in n_colors_list]

        if self._correction_processor is not None:
            correction_processor, correction_parameters = self._correction_processor
            cdata = correction_processor.corrected_values(correction_parameters,
                                                          x1=x, x2=x + 1, y1=y, y2=y + 1, z1=z, z2=z + 1)
            self._ax[1, 1].scatter(correction_processor.predictors[:, 0], cdata[:, 0, 0, 0], marker='o', s=3)

        for processor, prediction_parameters, correction_parameters, label in self._processors:
            cdata = processor.corrected_values(correction_parameters,
                                               x1=x, x2=x + 1, y1=y, y2=y + 1, z1=z, z2=z + 1
                                               )
            self._ax[1, 1].scatter(processor.predictors[:, 0], cdata[:, 0, 0, 0], marker='o', c=colors[0], s=3 )

            axis, curve = processor.curve(prediction_parameters,
                                          x1=x, x2=x + 1, y1=y, y2=y + 1, z1=z, z2=z + 1, tpoints=self._num_points
                                          )

            self._ax[1, 1].plot(axis.T, curve[:, 0, 0, 0],
                                label=label, color=colors[0])

            colors.append(colors[0])
            del colors[0]

        if len(self._processors) > 0 or hasattr(self, '_correction_processor'):
            self._ax[1, 1].legend(loc=1, fontsize='xx-small')

        # Set same aspect
        asp = np.diff(self._ax[1, 1].get_xlim())[0] / np.diff(self._ax[1, 1].get_ylim())[0]
        self._ax[1, 1].set_aspect(asp)
        self._ax[1, 1].grid()
        self._ax[1, 1].tick_params(direction='in')
        for item in self._ax[1, 1].get_xticklabels():
            item.set_fontsize(8)
        for item in self._ax[1, 1].get_yticklabels():
            item.set_fontsize(8)


        # Compute mm coordinates with affine
        c_voxel = [new_voxel[0], new_voxel[1], new_voxel[2], 1]
        c_voxel = np.array(c_voxel, dtype=np.float32)
        mm_coords = self._affine.dot(c_voxel)[:-1]

        atlas_name_list = [atlas_dict[atlas[x,y,z]] for atlas, atlas_dict in zip(self._atlas, self._atlas_dict)]

        self._figure.canvas.set_window_title(' - '.join(atlas_name_list) + ' - Coordinates {}, {}, {}'.format(*mm_coords))

        self._figure.canvas.draw()

        self._current_voxel = new_voxel

    def show(self):
        self._figure = plt.figure()

        outer_padding = (0.04, 0.04, 0.04, 0.04)  # left, right, bottom, up
        inner_padding = (0.02, 0.02)  # horizontal, vertical

        total_width = self._template_shape[0] + self._template_shape[1]
        total_height = self._template_shape[2] + self._template_shape[1]

        effective_width = 1. - outer_padding[0] - outer_padding[1] - inner_padding[0]
        effective_height = 1. - outer_padding[2] - outer_padding[3] - inner_padding[1]

        width1 = self._template_shape[0] * effective_width / total_width
        width2 = effective_width - width1

        height1 = self._template_shape[2] * effective_height / total_height
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

            if cmap != 'rgb':
                cmap = cm.get_cmap(cmap)
                img = cmap(img)

            Cfg, Afg = img[:, :, :, :3], img[:, :, :, 3]
            Cbg, Abg = self._rgba_image[:, :, :, :3], self._rgba_image[:, :, :, 3]

            Ar = Afg + Abg * (1 - Afg)
            Cr = np.zeros(Cbg.shape)
            for c in range(Cr.shape[3]):
                Cr[:, :, :, c] = Cfg[:, :, :, c] * Afg + Cbg[:, :, :, c] * Abg * (1 - Afg) / Ar

            self._rgba_image[:, :, :, :3] = Cr
            self._rgba_image[:, :, :, 3] = Ar

        for mask in self._mask:
            for i in range(self._rgba_image.shape[-1]):
                self._rgba_image[..., i] = self._rgba_image[..., i] * mask

        self._current_voxel = 0, 0, 0

        self._figure.canvas.mpl_connect('button_press_event', self.__button_press_event__)
        self._figure.canvas.mpl_connect('motion_notify_event', self.__motion_notify_event__)

        # if len(self._images) > 0:
        #     new_voxel = [index_where[0] for index_where in np.where(self._images[-1][0] == False)]
        # else:
        new_voxel = list(map(lambda x: int(x / 2), self._template_shape))


        self.__update_views__(new_voxel)

        plt.show()


class GUIVisualizer_surface(object):
    def __init__(self, template, affine, num_points=100, template_cmap='gray'):
        self._template = template
        if not isinstance(template, tuple):
            raise ValueError('Please, specify a proper template with extension [\'inflated\', \'pial\', \'white\'].')

        self._template_shape = template[0].shape[0]
        self._affine = affine
        self._num_points = num_points
        self._template_cmap = template_cmap
        self._images = []
        self._processors = []
        self._correction_processor = None
        self._figure = None
        self._ax = None
        self._rgba_image = None
        self._current_vertex = None

    def add_image(self, image, colormap='hot'):
        """
        Remember that you can mask the image before adding it to the plot as follows:
        For a case in which we would like to make transparent the vertexs of the matrix that contain a value of 0,
        assuming that original_image is a numpy (np) array:
        masked_image = np.ma.masked_where(original_image == 0, original_image))

        To make transparent any other vertexs, change 'original_image == 0' in the previous line with the condition
        that the vertexs to be masked must fulfill
        """
        if image.shape[0] != self._template_shape:
            raise ValueError("The shape of the image " + str(image.shape) + " must match that of the template " + str(
                self._template_shape))

        # image = (image - np.min(image)).astype(float) / (np.max(image) - np.min(image))

        self._images.append((image, colormap))
        return self

    def add_curve_processor(self, processor, prediction_parameters, correction_parameters, label=None):
        if prediction_parameters.shape[1] != self._template_shape:
            raise ValueError("The shape of the prediction parameters " + str(
                prediction_parameters.shape[1:]) + " must match that of the template " + str(self._template_shape))
        if correction_parameters.shape[1] != self._template_shape:
            raise ValueError("The shape of the correction parameters {} must match that of the template {}".format(
                correction_parameters.shape[1:],
                self._template_shape
            ))
        if label is None:
            label = 'Curve {}'.format(len(self._processors) + 1)
        self._processors.append((processor, prediction_parameters, correction_parameters, label))
        return self

    def set_corrected_data_processor(self, processor, correction_parameters):
        if correction_parameters.shape[1:] != self._template_shape:
            raise ValueError("The shape of the correction parameters " + str(
                correction_parameters.shape[1:]) + " must match that of the template " + str(self._template_shape))

        self._correction_processor = (processor, correction_parameters)

    def __compute_new_vertex_coords__(self, event):
        if event.inaxes is self._ax[0]:
            new_vertex = event.xdata
        else:
            return None

        return tuple(int(0.5 + new_vertex[i]) for i in range(len(new_vertex)))

    def __compute_xydata__(self, vertex):
        xydata00 = vertex#, (self._template_shape[1] - 1) - vertex[1]
        shape00 = self._template_shape#, self._template_shape[1]

        return (xydata00, shape00)

    def __button_press_event__(self, event):
        if event.button != 1:
            return

        self.__update_views__(self.__compute_new_vertex_coords__(event))

    def __motion_notify_event__(self, event):
        if event.button != 1:
            return

        self.__update_views__(self.__compute_new_vertex_coords__(event))

    def __image_cut__(self, vertex):
        return list(self._template)

    def __update_views__(self, new_vertex):
        if new_vertex is None or new_vertex == self._current_vertex:
            return

        axes = (self._ax[0], self.__image_cut__, cm.get_cmap(self._template_cmap))
        # xydata = self.__compute_xydata__(new_vertex)

        for i in range(len(self._images)):
            ax, bg_cut, bg_cmap = axes
            fg_cut, fg_cmap = self._images[i]
            ax.clear()

            plotting.plot_surf(bg_cut(new_vertex), surf_map = fg_cut, cmap=fg_cmap, axes=ax)
            #
            # (xdata,), (width, height) = xydata
            # ax.add_line(mplline(xdata=[0, width], ydata=[ydata, ydata], linewidth=1, color='green'))
            # ax.add_line(mplline(xdata=[xdata, xdata], ydata=[0, height], linewidth=1, color='green'))

        self._ax[1].clear()

        x = new_vertex

        colors = ['r', 'c', 'g', 'm', 'y', 'k']

        if self._correction_processor is not None:
            correction_processor, correction_parameters = self._correction_processor
            cdata = correction_processor.corrected_values(correction_parameters,x1=x, x2=x + 1)
            self._ax[1].plot(correction_processor.predictors[:, 0], cdata[:, 0], 'bo')

        for processor, prediction_parameters, correction_parameters, label in self._processors:
            cdata = processor.corrected_values(correction_parameters,x1=x, x2=x + 1)
            self._ax[1].plot(processor.predictors[:, 0], cdata[:, 0], 'bo', color=colors[0])

            axis, curve = processor.curve(prediction_parameters,x1=x, x2=x + 1, tpoints=self._num_points )
            self._ax[1].plot(axis.T, curve[:, 0],label=label, color=colors[0], marker='d')

            colors.append(colors[0])
            del colors[0]

        if len(self._processors) > 0 or hasattr(self, '_correction_processor'):
            self._ax[1].legend()

        # Compute mm coordinates with affine
        # c_vertex = [new_vertex, 1]
        # c_vertex = np.array(c_vertex, dtype=np.float32)
        # mm_coords = self._affine.dot(c_vertex)[:-1]
        # self._figure.canvas.set_window_title('Coordinates {}, {}, {}'.format(*mm_coords))

        self._figure.canvas.draw()

        self._current_vertex = new_vertex

    def show(self):
        self._figure = plt.figure()

        outer_padding = (0.04, 0.04, 0.04, 0.04)  # left, right, bottom, up
        inner_padding = (0.02, 0.02)  # horizontal, vertical

        # total_width = self._template_shape[0]# + self._template_shape[1]
        # total_height = self._template_shape[0]# + self._template_shape[1]

        effective_width = 1. - outer_padding[0] - outer_padding[1] - inner_padding[0]
        effective_height = 1. - outer_padding[2] - outer_padding[3] - inner_padding[1]

        # width1 = self._template_shape[0] * effective_width / total_width
        # width2 = effective_width - width1
        #
        # height1 = self._template_shape[0] * effective_height / total_height
        # height2 = effective_height - height1

        self._ax = np.array([
            self._figure.add_subplot(2, 1, 1, projection='3d'),
            self._figure.add_subplot(2, 1, 2),
        ])

        self._ax[0].set_xticks([])
        self._ax[0].set_yticks([])

        #
        # for img, cmap in self._images:
        #     cmap = cm.get_cmap(cmap)
        #     img = cmap(img)
        #
        #     Cfg, Afg = img[:, :, :, :3], img[:, :, :, 3]
        #     Cbg, Abg = self._rgba_image[:, :, :, :3], self._rgba_image[:, :, :, 3]
        #
        #     Ar = Afg + Abg * (1 - Afg)
        #     Cr = np.zeros(Cbg.shape)
        #     for c in range(Cr.shape[3]):
        #         Cr[:, :, :, c] = Cfg[:, :, :, c] * Afg + Cbg[:, :, :, c] * Abg * (1 - Afg) / Ar
        #
        #     self._rgba_image[:, :, :, :3] = Cr
        #     self._rgba_image[:, :, :, 3] = Ar

        self._current_vertex = 0

        self._figure.canvas.mpl_connect('button_press_event', self.__button_press_event__)
        self._figure.canvas.mpl_connect('motion_notify_event', self.__motion_notify_event__)

        new_vertex = self._template_shape/2

        self.__update_views__(new_vertex)

        plt.show()


class GUIVisualizer_latent(GUIVisualizer):

    def __init__(self, template, affine, predictor_names, num_points=100, template_cmap='gray'):
        self.predictor_names = predictor_names
        super(GUIVisualizer_latent, self).__init__(template, affine, num_points=num_points, template_cmap=template_cmap)


    def __update_views__(self, new_voxel):
        if new_voxel is None or new_voxel == self._current_voxel:
            return

        axes = [(self._ax[0, 0], self.__coronal_cut__), (self._ax[0, 1], self.__sagittal_cut__),
                (self._ax[1, 0], self.__axial_cut__)]
        xydata = self.__compute_xydata__(new_voxel)
        for i in range(len(axes)):
            ax, cut = axes[i]
            ax.clear()
            ax.imshow(cut(new_voxel), interpolation='nearest')

            (xdata, ydata), (width, height) = xydata[i]
            ax.add_line(mplline(xdata=[0, width], ydata=[ydata, ydata], linewidth=1, color='green'))
            ax.add_line(mplline(xdata=[xdata, xdata], ydata=[0, height], linewidth=1, color='green'))

        self._ax[1, 1].clear()

        x, y, z = new_voxel

        colors = ['r', 'c', 'g', 'm', 'y', 'k']

        it_proc = 0
        M = 0
        for processor, prediction_parameters, correction_parameters, label in self._processors:
            x_rotations = processor.prediction_processor.fitter.get_item_parameters(prediction_parameters[:,x:x+1,y:y+1,z:z+1],
                                                                                    name='x_rotations')
            if hasattr(x_rotations,'shape'):
                M = x_rotations[0].shape[0]
                for it_nc in range(x_rotations.shape[0]):
                    print(np.squeeze(x_rotations[it_nc]).shape)
                    self._ax[1, 1].barh(np.arange(M) + 0.2*it_proc, np.squeeze(x_rotations[it_nc]), align='center',
                                        color = colors[it_proc], ecolor='black')

                    it_proc +=1


        ##### NEW
        self._ax[1, 1].set_yticks(np.arange(start=0, stop=M, step=1))
        self._ax[1, 1].set_yticklabels(self.predictor_names)
        self._ax[1, 1].invert_yaxis()  # labels read top-to-bottom
        self._ax[1, 1].grid()
        self._ax[1, 1].set_xlim([-1, 1])

        #####


        if len(self._processors) > 0 or hasattr(self, '_correction_processor'):
            self._ax[1, 1].legend()

        # Compute mm coordinates with affine
        c_voxel = [new_voxel[0], new_voxel[1], new_voxel[2], 1]
        c_voxel = np.array(c_voxel, dtype=np.float32)
        mm_coords = self._affine.dot(c_voxel)[:-1]
        self._figure.canvas.set_window_title('Coordinates {}, {}, {}'.format(*mm_coords))

        self._figure.canvas.draw()

        self._current_voxel = new_voxel
