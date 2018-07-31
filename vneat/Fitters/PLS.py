import numpy as np
from sklearn.cross_decomposition import PLSRegression
import warnings
from vneat.Fitters.CurveFitting import AdditiveCurveFitter


class PLS(AdditiveCurveFitter):
    ''' Class that implements the Projection to Latent Structures method.

        This method assumes the following situation:

            - There are M (random) variables whose behaviour we want to explain.
              Each one of them behave independently of the other, hence, a PLS
              model is build for each one of them

            - Each of the M variables has been measured N times, obtaining thus
              an NxM matrix of observations (i-th column contains the N observa-
              tions for i-th variable).

            - There are K predictors (in this class both, the correctors and the
              predictors are called predictors and treated equally) that might
              explain the behaviour of the M variables in an additive manner, i.e.,
              a ponderated sum of the K predictors might fit each of the variables.

            - Each of the K predictors has been measured at the same moments in
              which the M variables were measured, giving thus a NxK matrix where
              the i-th column represents the N observations of the i-th predictor.
        

    '''

    def __init__(self, num_components_corr=None, num_components_pred=None, predictors=None, correctors=None):

        self.num_components_corr = num_components_corr
        self.num_components_pred = num_components_pred

        super(PLS, self).__init__(predictors=predictors, correctors=correctors)


    def __predict__(self, predictors, prediction_parameters, *args, **kwargs):
        '''Computes a prediction applying the prediction function used in GLM.

            Parameters:

                - predictors: NxR (2-dimensional) matrix, representing the predictors, i.e., features to be used
                    to try to explain/predict the observations (experimental data), where R is the number of
                    predictors and N the number of elements for each predictor.

                - prediction_parameters: RxM (2-dimensional) matrix, representing the parameters that best fit
                    the predictors to the corrected observations for each variable, where M is the number of
                    variables and K is the number of prediction parameters for each variable.

                - curve_bool: bool. It refers whether to display a curve (hence R curves) or to just predict the regression
                    equation.

                - any other arguments will also be passed to the method in the subclass.

            Returns:

                - Prediction: NxM (2-dimensional) matrix, containing N predicted values for each of the M variables,
                    result of computing the expression 'predictors * prediction_parameters' (matrix multiplication).
        '''

        curve_bool = kwargs.pop('curve', False)
        num_components = PLS.get_num_components(prediction_parameters)
        y_mean = prediction_parameters[-3]

        print(curve_bool)
        if num_components == 0:
            if curve_bool:
                if len(y_mean.shape) == 1:
                    y_mean = y_mean[np.newaxis, np.newaxis, :]

                return np.repeat(np.repeat(y_mean/predictors.shape[1],predictors.shape[1],axis=1),predictors.shape[0],axis=0)
            else:
                return y_mean
        else:
            if curve_bool:
                x_mean = PLS.get_x_mean(prediction_parameters)
                x_coef = PLS.get_x_coef(prediction_parameters)

                if len(y_mean.shape) < 3:
                    y_mean = y_mean[np.newaxis, np.newaxis, :]

                y_mean = np.repeat(np.repeat(y_mean / predictors.shape[1], predictors.shape[1], axis=1), predictors.shape[0], axis=0)
                return np.multiply((predictors-x_mean)[..., np.newaxis],x_coef)  + y_mean

            else:

                x_coef = PLS.get_x_coef(prediction_parameters)
                x_mean = PLS.get_x_mean(prediction_parameters)


                predictors -= x_mean


                return np.dot(predictors,x_coef) + y_mean

    def __fit__(self,correctors, predictors, observations, n_jobs=-1, *args, **kwargs):
        '''Computes the correction and prediction parameters that best fit the observations according to the
            Partial Least Squares metdhos

            Parameters:

                - correctors: NxC (2-dimensional) matrix, representing the covariates, i.e., features that
                    (may) explain a part of the observational data in which we are not interested, where C
                    is the number of correctors and N the number of elements for each corrector.

                - predictors: NxR (2-dimensional) matrix, representing the predictors, i.e., features to be used
                    to try to explain/predict the observations (experimental data), where R is the number of
                    predictors and N the number of elements for each predictor (the latter is ensured to be the
                    same as that in the 'correctors' argument).

                - observations: NxM (2-dimensional) matrix, representing the observational data, i.e., values
                    obtained by measuring the variables of interest, whose behaviour is wanted to be explained
                    by the correctors and predictors, where M is the number of variables and N the number of
                    observations for each variable (the latter is ensured to be the same as those in the
                    'correctors' and the 'predictors' arguments).


                - num_threads: integer (default -1), indicating the number of threads to be used by the algo-
                    rithm. If set to -1, all CPUs are used. This will only provide speed-up for M > 1 and
                    sufficiently large problems.

            Returns:

                - Correction parameters: (num_comp+2)*CxM (3-dimensional) matrix, representing the parameters that best fit
                    the correctors to the observations for each variable, where M is the number of variables
                    (same as that in the 'observations' argument) and C is the number of correction parameters
                    for each variable (same as the number of correctors).

                - Regression parameters: ((num_comp+2)*R + 2)xM (3-dimensional) matrix, representing the parameters that best fit
                    the predictors to the corrected observations for each variable, where M is the number of
                    variables (same as that in the 'observations' argument) and R is the number of prediction
                    parameters for each variable (same as the number of predictors).
                    The first dimension correspond to (x_rotations, coef, x_mean, y_mean, num_components)
        '''

        # All-at-once approach
        pls_corr = PLSRegression(n_components=self.num_components_corr, scale=False)
        pls_pred = PLSRegression(n_components=self.num_components_pred, scale=False)

        M = observations.shape[1]
        R = predictors.shape[1]

        if correctors.size != 0:
            cparams = np.zeros((R*(self.num_components_pred+2)+3,M))
            for n in range(M):
                if np.std(observations[:, n]) == 0:
                    continue
                pls_corr.fit(correctors, observations[:, n])
                observations[:, n] = observations[:, n] - np.dot(pls_corr.transform(correctors), pls_corr.y_loadings_.T)

                cparams[:R * self.num_components_corr, n] = pls_corr.x_rotations_.reshape((-1,))
                cparams[R * self.num_components_corr:R * (self.num_components_corr + 1), n] = pls_corr.coef_.reshape((-1,))
                cparams[R * (self.num_components_corr + 1):-2, n] = pls_corr.x_mean_.reshape((-1,))
                cparams[-3, n] = pls_corr.y_mean_.reshape((-1,))
                cparams[-2, n] = correctors.shape[1]
                cparams[-1, n] = self.num_components_corr
                cparams = np.concatenate((pls_corr.x_rotations_[np.newaxis],pls_corr.y_loadings_[np.newaxis],
                                          pls_corr.x_mean_[np.newaxis], pls_corr.y_mean_[np.newaxis]),axis=0)
        else:
            cparams = np.asarray([[]])

        if predictors.size != 0:
            pparams = np.zeros(((R+1)*(self.num_components_pred+1)+R+2,M))
            for n in range(M):
                if np.std(observations[:,n]) == 0:
                    pparams[-3, n] = np.mean(observations[:,n]).reshape((-1,))
                    continue
                pls_pred.fit(predictors,observations[:,n])
                pparams[:R*self.num_components_pred,n] = pls_pred.x_rotations_.reshape((-1,))
                pparams[R*self.num_components_pred:(R+1)*self.num_components_pred ,n] = pls_pred.y_rotations_.reshape((-1,))
                pparams[(R+1)*self.num_components_pred:(R+1)*self.num_components_pred+R,n] = pls_pred.coef_.reshape((-1,))
                pparams[(R+1)*self.num_components_pred+R:-3,n] = pls_pred.x_mean_.reshape((-1,))
                pparams[-3,n] = pls_pred.y_mean_.reshape((-1,))
                pparams[-2,n] = R
                pparams[-1,n] = self.num_components_pred

        else:
            pparams = np.asarray([[]])

        return (cparams,pparams)

    def __transform__(self, predictors, prediction_parameters, observations = None, *args, **kwargs):

        num_components = PLS.get_num_components(prediction_parameters)
        y_mean = PLS.get_y_mean(prediction_parameters)#prediction_parameters[-2]
        num_subjects = predictors.shape[0]
        N = np.prod(prediction_parameters.shape[1:])

        if num_components == 0:
            if observations is not None:
                observations -= y_mean
                return np.zeros((num_components, num_subjects, N)), np.zeros((num_components, num_subjects, N))
            else:
                return np.zeros((num_components, num_subjects, N))

        x_rotations = PLS.get_x_rotations(prediction_parameters)#prediction_parameters[:R*num_components].reshape((R,num_components,-1))
        x_mean = PLS.get_x_mean(prediction_parameters)#prediction_parameters[(R+1)*(num_components+1):-2,0]

        predictors -= x_mean
        x_scores = np.zeros((num_components, num_subjects, N))
        for it_nc in range(num_components):
            x_scores[it_nc] = np.dot(predictors, x_rotations[it_nc])

        if observations is not None:
            observations -= y_mean
            y_rotations = PLS.get_y_rotations(prediction_parameters)#1 * np.ones((num_components,)+prediction_parameters.shape[1:])  # prediction_parameters[R*num_components:(R+1)*num_components].reshape((1,num_components, -1))

            y_scores = np.zeros((num_components, num_subjects, N))
            for it_nc in range(num_components):
                y_scores[it_nc] = np.multiply(observations, y_rotations[it_nc, :])


            return x_scores, y_scores

        else:

            return x_scores

    def __df_correction__(self, observations, correctors, correction_parameters):
        # It computes (naively yet) DoF of the method, equal to the number of correction latent variables
        warnings.warn('DoF for PLS naively computed ...for more accurate implementation see '
                      'Kramer et. al: The Degrees of Freedom of Partial Least Squares Regression')
        return np.ones((1, observations.shape[1])) * self.num_components_corr

    def __df_prediction__(self, observations, predictors, prediction_parameters):
        # It computes (naively yet) DoF of the method, equal to the number of prediction latent variables
        warnings.warn('DoF for PLS naively computed ...for more accurate implementation see '
                      'Kramer et. al: The Degrees of Freedom of Partial Least Squares Regression')
        return np.ones((1, observations.shape[1])) * self.num_components_pred


    def get_item_parameters(self, parameters, name = None):
        if name =='x_rotations':
            return PLS.get_x_rotations(parameters)
        elif name == 'x_coef':
            return PLS.get_x_coef(parameters)
        elif name =='num_components':
            return PLS.get_num_components(parameters)
        else:
            raise ValueError('There is no parameter ' + str(name) + ' in this fitter.')

    @staticmethod
    def get_x_rotations(parameters):
        ''' Returns x_rotation from parameters of Fitters.PLS.PLS model

            Parameters:

                - parameters: ((R + 1) * L + R + 3) x X1 x X2, ...x Xr array, where
                    L=num_components, R=num_predictors.

            Returns:

                - x_rotations: L x R x X1, X2, ..., Xr array NEED CHECKING!!!
        '''
        num_components = int(np.unique(np.sort(parameters[-1]))[-1])
        R = int(np.unique(np.sort(parameters[-2]))[-1])
        dims = parameters.shape[1:]


        if num_components == 0:
            return np.zeros((num_components,R) + dims)
        else:
            x_rotations_tmp = parameters[:R * num_components].reshape((R,num_components,) + dims)
            x_rotations = np.zeros((num_components,R) + dims)
            for it_nc in range(num_components):
                x_rotations[it_nc] = x_rotations_tmp[:,it_nc]

            return x_rotations

    @staticmethod
    def get_x_coef(parameters):

        R = int(np.unique(np.sort(parameters[-2]))[-1])#3
        num_components = int(np.unique(np.sort(parameters[-1]))[-1])

        return parameters[R * num_components:R * num_components + R]

    @staticmethod
    def get_x_mean(parameters):
        num_components = int(np.unique(np.sort(parameters[-1]))[-1])
        R = int(np.unique(np.sort(parameters[-2]))[-1])#3
        return parameters[(R + 1) * num_components + R :-3, 0]

    @staticmethod
    def get_y_mean(parameters):
        return parameters[-3]

    @staticmethod
    def get_y_rotations(parameters):
        num_components = int(np.unique(np.sort(parameters[-1]))[-1])
        R = int(np.unique(np.sort(parameters[-2]))[-1])#3
        dims = parameters.shape[1:]

        if num_components == 0:
            return np.zeros((num_components, R) + dims)
        else:
            return parameters[R * num_components : (R+1) * num_components].reshape((num_components,)+dims)

    @staticmethod
    def get_num_components(parameters):
        return int(np.unique(np.sort(parameters[-1]))[-1])
