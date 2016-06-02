import types

import numpy as np
from scipy.stats import f as f_stat

class UNBOUND(object):
    pass

class dictlike(type):
    class _bound_generic_evaluation_function(object):
        def __init__(self, target):
            self._target = target
            self._set = []

        @property
        def target(self):
            return self._target
        
        def bind(self, method_name, method):
            setattr(self, method_name, types.MethodType(method, self))
            if not method_name in self._set:
                self._set.append(method_name)
            return self

        def unbind(self, method_name):
            for i in xrange(len(self._set)):
                if self._set[i] == method_name:
                    del self._set[i]
                    exec('del self.' + method_name)
                    return self

            raise AttributeError("Method '" + method_name + "' was not bound to this target and cannot be unbound from it.")

        def clear(self):
            for name in self._set:
                exec('del self.' + name)
            self._set = []
            return self

        def status(self):
            s = 'Target:\n    ' + str(self._target) + '\n\n'
            s += 'Description:\n    Test-generic requirement lookup set.\n\n'
            s += 'Methods:\n'

            for name in self._set:
                s += '        [Bound]  ' + name + '\n'
            s += '\n'

            return s


    def __init__(self, *args, **kwargs):
        super(dictlike, self).__init__(*args, **kwargs)
        self._bindings = {}

    def __getitem__(self, target):
        try:
            return self._bindings[target]
        except KeyError:
            retval = dictlike._bound_generic_evaluation_function(target)
            self._bindings[target] = retval
            return retval

    def clear(self):
        self._bindings = {}
        return self

class evaluation_function(object):

    __metaclass__ = dictlike

    class RequirementDescriptor:
        def __init__(self, name, description, default = UNBOUND):
            self._name = name
            self._description = description
            self._value = default
        
        @property
        def name(self):
            return self._name
        
        @property
        def description(self):
            return self._description

        @property
        def value(self):
            return self._value


    class _bound_evaluation_function(object):
        def __init__(self, parent, target, eval_func, implicit, requirements):
            self._parent = parent
            self._target = target
            self._evaluate = eval_func
            self._requirements = [evaluation_function.RequirementDescriptor(name, desc, UNBOUND) for (name, desc) in requirements.iteritems()]
            self._implicit = map(lambda rd: evaluation_function.RequirementDescriptor(rd.name, rd.description, types.MethodType(rd.value, self)), implicit.itervalues())

            self._forced = []

            self.clear()

        @property
        def target(self):
            return self._target
        
        @property
        def requirements(self):
            return self._requirements
        
        @property
        def implicit(self):
            return self._implicit

        def _inherit(self, status=False):
            try:
                mro = self._target.mro()[1:] # the target is analyzed separately
            except AttributeError:
                mro = type(self._target).mro()
            except AttributeError:
                mro = [] # [self._target] if isinstance(self._target, type) else [type(self._target)]

            inherited = []

            # Check whether the requirements have been fulfilled, and inherit those that have not been explicitly bound
            for ReqDescriptor in self._requirements:
                # First step, check whether the method was explicitly bound for this test and target
                if getattr(self, ReqDescriptor.name) is UNBOUND:
                    # Seems like it wasn't
                    # Check whether a test-generic method for this class was defined specifically
                    try:
                        provider = evaluation_function._bindings[self._target]

                        # Woo-hoo! There is at least one generic method declared explicitly for the target
                        inherited_method = getattr(provider, ReqDescriptor.name)

                        # Bingo! Our current requirement IS generically specified for the target
                        setattr(self, ReqDescriptor.name, inherited_method)
                        inherited.append(ReqDescriptor.name)
                        continue
                    except (KeyError, AttributeError):
                        pass

                    # Check each ancestor (in order), to see whether they explicitly bound the required method for this test
                    # or if, at least, they declared such method generally for all tests.
                    for i in xrange(len(mro)):
                        found = False
                        try:
                            provider = self._parent._bindings[mro[i]]

                            # no need to check for AttributeError, given that all the targets share the same requirements
                            # (set to UNBOUND in the worst case)
                            inherited_method = getattr(provider, ReqDescriptor.name)
                            if not (inherited_method is UNBOUND):
                                found = True
                        except KeyError:
                            pass

                        if not found:
                            # test-specific method lookup failed, try test-generic method lookup
                            try:
                                provider = evaluation_function._bindings[mro[i]]

                                # Woo-hoo! There is at least one generic method declared explicitly for this class
                                inherited_method = getattr(provider, ReqDescriptor.name)

                                # Bingo! Our current requirement IS generally specified for this class
                            except (KeyError, AttributeError):
                                continue

                        # Yay, we found a method to inherit!
                        setattr(self, ReqDescriptor.name, inherited_method)
                        inherited.append(ReqDescriptor.name)
                        break
                    else: # bad news... :(
                        if not status: # ...unless we are checking the status, in which case there's no problem
                            # rollback
                            self._revert_inheritance(inherited)

                            # and raise
                            raise RuntimeError("Method '" + ReqDescriptor.name + "' could not be inherited.")

            # everything went smoothly
            return inherited

        def _revert_inheritance(self, inherited):
            for name in inherited:
                setattr(self, name, UNBOUND)

        def bind(self, method_name, method, force=False):
            reqs = map(lambda rd: rd.name, self._requirements + self._implicit)
            if not (method_name in reqs):
                if force:
                    self._forced.append(method_name)
                else:
                    raise AttributeError("Method '" + method_name + "' was not defined as a requirement for this test and can not be bound to it.")
            
            setattr(self, method_name, types.MethodType(method, self))
            return self

        def unbind(self, method_name):
            reqs = self._requirements + self._implicit
            for rd in reqs:
                if rd.name == method_name:
                    setattr(self, rd.name, rd.value)
                    try:
                        self._frozen.remove(rd.name)
                    except ValueError:
                        pass

                    return self
            for i in xrange(len(self._forced)):
                if self._forced[i] == method_name:
                    exec('del self.' + method_name)
                    del self._forced[i]
                    return self

            raise AttributeError("Method '" + method_name + "' was not bound for this test and can not be unbound from it.")

        def clear(self):
            for ReqDescriptor in self._implicit:
                setattr(self, ReqDescriptor.name, ReqDescriptor.value)
            for ReqDescriptor in self._requirements:
                setattr(self, ReqDescriptor.name, ReqDescriptor.value)
            self._frozen = []

            for name in self._forced:
                exec('del self.' + name)

            self._forced = []

            return self

        def evaluate(self, fitting_results = None, *args, **kwargs):
            if not (fitting_results is None):
                self.fitting_results = fitting_results
            inherited = self._inherit()

            # all clear, do the job!
            try:
                retval = self._evaluate(self, *args, **kwargs)
            except Exception as e:
                # oops! something went wrong; rollback and re-raise
                self._revert_inheritance(inherited)
                raise e

            # everything fine, undo inheritance and return the result
            self._revert_inheritance(inherited)

            if not (fitting_results is None):
                del self.fitting_results

            return retval

        def freeze(self):
            '''Stops dynamic inheritance; takes a snapshot of the current status and stores it.
            '''
            inherited = self._inherit()
            self._frozen.extend(inherited)

            return self

        def unfreeze(self):
            '''Restarts dynamic inheritance.
            '''
            self._revert_inheritance(self._frozen)
            self._frozen = []

            return self

        def status(self):
            desc = {}
            for rd in self._implicit:
                desc[rd.name] = '      [Default]  ' + rd.name + ': ' + rd.description
            for rd in self._requirements:
                desc[rd.name] = rd.name + ': ' + rd.description

            reqs = set(map(lambda r: r.name, self._requirements))

            inherited = self._inherit(status=True)
            for name in inherited:
                desc[name] = '    [Inherited]  ' + desc[name]
                reqs.remove(name)
            for name in self._frozen:
                desc[name] = '       [Frozen]  ' + desc[name]
                reqs.remove(name)

            for name in reqs:
                if getattr(self, name) is UNBOUND:
                    desc[name] = '     [Required]  ' + desc[name]
                else:
                    desc[name] = '        [Bound]  ' + desc[name]

            self._revert_inheritance(inherited)

            s = 'Target:\n    ' + str(self._target) + '\n\n'
            s += 'Description:\n    ' + ('None\n' if self._evaluate.__doc__ is None else self._evaluate.__doc__) + '\n'
            s += 'Methods:\n'
            for rd in self._implicit:
                s += desc[rd.name] + '\n'
            for rd in self._requirements:
                s += desc[rd.name] + '\n'
            for name in self._forced:
                s += '       [Forced]  ' + name + ': ' + ('None\n' if getattr(self, name).__doc__ is None else getattr(self, name).__doc__) + '\n'
            s += '\n'

            return s


    def __init__(self, func):
        self._bindings = {}
        self._evaluate = func
        self._requirements = {}
        self._implicit = {}

        self.clear = self._clear

    def requires(self, method_name, description):
        """Specifies that a method whose name is contained in 'method_name' argument is necessary to evaluate this test.
            The description of such method is given in 'description' argument.
        """
        if len(self._bindings) > 0:
            raise RuntimeError('This function has already been bound at least once; it is not possible to specify more requirements at this point.')
        
        self._requirements[method_name] = description

        return self

    def implicit(self, method_name, description, bound_method):
        """Specifies the name, description and value of a method that is required by the test to be evaluated.
            In this case, any target will call the method contained in 'bound_method' argument unless it has been
            're-bound' to another method for that specific target.
            IMPORTANT: methods bound this way take preference over inheritance.
        """
        if len(self._bindings) > 0:
            raise RuntimeError('This function has already been bound at least once; it is not possible to specify more default methods at this point.')
        
        try:
            del self._requirements[method_name]
        except KeyError:
            pass

        self._implicit[method_name] = evaluation_function.RequirementDescriptor(method_name, description, bound_method)
        
        return self

    def __getitem__(self, target):
        try:
            return self._bindings[target]
        except KeyError:
            retval = evaluation_function._bound_evaluation_function(
                parent = self,
                target = target,
                eval_func = self._evaluate,
                implicit = self._implicit,
                requirements = self._requirements
            )
            self._bindings[target] = retval
            return retval

    def _clear(self):
        self._bindings = {}
        return self


@evaluation_function
def mse(self):
    """Evaluates the significance of the predictors as regards the behaviour of the observations by computing
        the Mean Squared Error of the prediction with respect to the corrected data. The smaller the result,
        the better the fit.
    """
    # prediction_error = corrected_data - prediction
    prediction_error = self.corrected_data() - self.predicted_data()

    return (prediction_error**2).sum(axis = 0)/np.float64(len(prediction_error))

mse.requires('corrected_data', 'Matrix of shape (N, X1, ..., Xn) that contains the observations after having subtracted the contribution of the correctors, where N is the number of subjects/samples and M = X1*...*Xn the number of variables.')
mse.requires('predicted_data', 'Matrix of shape (N, X1, ..., Xn) that contains the prediction performed by the fitter on the corrected observations, where N is the number of subjects/samples and M = X1*...*Xn the number of variables.')


@evaluation_function
def r2(self):
    """Evaluates the significance of the predictors as regards the behaviour of the observations by computing
        the value of the R-squared measurement, which is basically a range adjusted version of the MSE.
        In this case, however, the larger the result, the better the fit.
    """
    corrected_data = self.corrected_data()
    correction_variance = ((corrected_data - corrected_data.mean(axis = 0))**2).sum(axis = 0)
    # We don't divide it by N-1 because the final ratio will eliminate this factor

    # prediction_error = corrected_data - prediction
    prediction_error = corrected_data - self.predicted_data()

    error_variance = ((prediction_error - prediction_error.mean(axis = 0))**2).sum(axis = 0)
    # We don't divide it by N-1 because the final ratio will eliminate this factor

    return 1 - error_variance / correction_variance

r2.requires('corrected_data', 'Matrix of shape (N, X1, ..., Xn) that contains the observations after having subtracted the contribution of the correctors, where N is the number of subjects/samples and M = X1*...*Xn the number of variables.')
r2.requires('predicted_data', 'Matrix of shape (N, X1, ..., Xn) that contains the prediction performed by the fitter on the corrected observations, where N is the number of subjects/samples and M = X1*...*Xn the number of variables.')


@evaluation_function
def fstat(self):
    """Evaluates the significance of the predictors as regards the behaviour of the observations by computing
        the value of the F-statistic for a test in which the null hypothesis states that the predictors do not
        explain the variation of the observations at all. The calculated F-statistic value compares the variance
        of the prediction error with the variance of the corrected data, WITHOUT then mapping the result to its
        corresponding p-value (which takes into account the degrees of freedom of both, the corrected data and
        the prediction error). Please, refer to the "ftest" method if what you wish is a p-value related measure
        rather than the F-statistic itself.
    """
    corrected_data = self.corrected_data()

    ## Get the error obtained when using the full model (correctors + predictors)
    # prediction = self.__predict__(predictors, prediction_parameters)

    # prediction_error = corrected_data - prediction
    prediction_error = corrected_data - self.predicted_data()

    ## Now compare the variances of the errors

    # Residual Sum of Squares for restricted model
    rss1 = ((corrected_data - corrected_data.mean(axis = 0))**2).sum(axis = 0)

    # Residual Sum of Squares for full model
    rss2 = (prediction_error**2).sum(axis = 0) # TODO: Check if this is correct or the following line should replace it
    # rss2 = ((prediction_error - prediction_error.mean(axis = 0))**2).sum(axis = 0)

    # Degrees of freedom
    dfc = self.df_correction()
    dfp = self.df_prediction()

    n = corrected_data.shape[0]
    df1 = dfp # degrees of freedom of rss1 - rss2
    df2 = n - dfc - dfp # degrees of freedom of rss2

    # Compute f-scores
    var1 = (rss1 - rss2)/df1
    var2 = rss2/df2
    f_score = var1/var2

    # print rss1, rss2
    # print 'Df Residuals:', df2
    # print 'Df Model:', df1
    # print 'F-statistic:', f_score
    # print 'R-squared:', 1 - rss2/rss1

    return f_score

fstat.requires('corrected_data', 'Matrix of shape (N, X1, ..., Xn) that contains the observations after having subtracted the contribution of the correctors, where N is the number of subjects/samples and M = X1*...*Xn the number of variables.')
fstat.requires('predicted_data', 'Matrix of shape (N, X1, ..., Xn) that contains the prediction performed by the fitter on the corrected observations, where N is the number of subjects/samples and M = X1*...*Xn the number of variables.')
fstat.requires('df_correction', 'Constant or matrix of shape (X1, ..., Xn) indicating the degrees of freedom of the correction model alone (without the predictors) for all variables (constant case) or each variable (matrix case).')
fstat.requires('df_prediction', 'Constant or matrix of shape (X1, ..., Xn) indicating the degrees of freedom of the prediction model alone (without the correctors) for all variables (constant case) or each variable (matrix case).')

@evaluation_function
def ftest(self):
    """Evaluates the significance of the predictors as regards the behaviour of the observations by performing
        an F-test. In particular, the null hypothesis states that the predictors do not explain the variation
        of the observations at all. The inverse of the p-value of such experiment (1 - p_value) is returned.
        Refer to the "fstats" method if what you are looking for is the value of the f-statistic rather than
        the p-value.
    """
    corrected_data = self.corrected_data()

    ## Get the error obtained when using the full model (correctors + predictors)
    # prediction = self.__predict__(predictors, prediction_parameters)

    # prediction_error = corrected_data - prediction
    prediction_error = corrected_data - self.predicted_data()

    ## Now compare the variances of the errors

    # Residual Sum of Squares for restricted model
    rss1 = ((corrected_data - corrected_data.mean(axis = 0))**2).sum(axis = 0)

    # Residual Sum of Squares for full model
    rss2 = (prediction_error**2).sum(axis = 0) # TODO: Check if this is correct or the following line should replace it
    # rss2 = ((prediction_error - prediction_error.mean(axis = 0))**2).sum(axis = 0)

    # Degrees of freedom
    dfc = self.df_correction()
    dfp = self.df_prediction()

    n = corrected_data.shape[0]
    df1 = dfp # degrees of freedom of rss1 - rss2
    df2 = n - dfc - dfp # degrees of freedom of rss2

    # Compute f-scores
    var1 = (rss1 - rss2)/df1
    var2 = rss2/df2
    f_score = var1/var2

    # Compute p-values
    return f_stat.cdf(f_score, df1, df2)

ftest.requires('corrected_data', 'Matrix of shape (N, X1, ..., Xn) that contains the observations after having subtracted the contribution of the correctors, where N is the number of subjects/samples and M = X1*...*Xn the number of variables.')
ftest.requires('predicted_data', 'Matrix of shape (N, X1, ..., Xn) that contains the prediction performed by the fitter on the corrected observations, where N is the number of subjects/samples and M = X1*...*Xn the number of variables.')
ftest.requires('df_correction', 'Constant or matrix of shape (X1, ..., Xn) indicating the degrees of freedom of the correction model alone (without the predictors) for all variables (constant case) or each variable (matrix case).')
ftest.requires('df_prediction', 'Constant or matrix of shape (X1, ..., Xn) indicating the degrees of freedom of the prediction model alone (without the correctors) for all variables (constant case) or each variable (matrix case).')


# TODO: Use alternative with 'uses' tool
#   @evaluation_function
#   def ftest(self):
#       """Evaluates the significance of the predictors as regards the behaviour of the observations by performing
#           an F-test. In particular, the null hypothesis states that the predictors do not explain the variation
#           of the observations at all. The inverse of the p-value of such experiment (1 - p_value) is returned.
#           Refer to the "fstats" method if what you are looking for is the value of the f-statistic rather than
#           the p-value.
#       """
#   
#   
#       try:
#           fitting_results = self.fitting_results
#       except AttributeError:
#           fitting_results = None
#   
#       f_score = fstat[self.target].evaluate(fitting_results)
#   
#       # Degrees of freedom
#       dfc = fstat[self.target].df_correction()
#       dfp = fstat[self.target].df_prediction()
#   
#       n = corrected_data.shape[0]
#       df1 = dfp # degrees of freedom of rss1 - rss2
#       df2 = n - dfc - dfp # degrees of freedom of rss2
#   
#       # Compute p-values
#       return f_stat.cdf(f_score, df1, df2)
#   
#   # TODO: make expression below (commented) work with all inheritance and stuff
#   # ftest.uses(fstat)

@evaluation_function
def aic(self):
    """Evaluates the significance of the predictors as regards the behaviour of the observations by computing
        the Akaike Information Criterion (AIC).
    """
    k = self.num_estimated_parameters()
    L = self.max_likelihood_value()

    return 2*k - 2*np.log(L)

aic.requires('num_estimated_parameters', 'The number of estimated parameters by the model (in total), counting the residual error as being one of them.')
aic.requires('max_likelihood_value', 'The maximum value that the likelihood function for this model can take.')


#   @evaluation_function
#   def aicc(correction_fitter, prediction_fitter, observations, correctors, correction_parameters, predictors, prediction_parameters):
#       """Evaluates the significance of the predictors as regards the behaviour of the observations by computing
#           the corrected Akaike Information Criterion (AICc).
#       """
#       k = correction_parameters.shape[0] + prediction_parameters.shape[0] + 1 # the residual error counts as an estimated variable
#       n = observations.shape[0]
#   
#       return aic(correction_fitter, prediction_fitter, observations, correctors, correction_parameters, predictors, prediction_parameters) + 2*k*(k+1)/np.float64(n - k - 1)
#   
#   
#   
#   
#   
#   # Curve based measures
#   
#   #TODO: create some methods to evaluate the fit based on the curve, the predictions and the corrected data
#   # One method: Penalized Residual Sum of Squares (PSRR) = MSE + lambda*sum(d2(curve)/d(x2))
#   # Another method: Modified Penalized Residual Sum of Squares (MPSRR) = MSE + lambda*max(d2(curve)/d(x2))
#   

@evaluation_function
def prss(self, gamma):
    """Evaluates the goodness of fit by means of the Penalized Residual Sum of Squares. In particular, this method
        computes the following expression: PRSS = MSE + gamma*sum(d2(curve)/d(x2)), that is, the Mean Squared Error
        plus a penalization parameter (gamma) times an indicator of the abruptness of the curve (i.e., the integral
        of the second derivative of the curve in the region of interest).
    """
    try:
        fitting_results = self.fitting_results
    except AttributeError:
        fitting_results = None

    MSE = mse[self.target].evaluate(fitting_results)
    curve = np.array(self.curve(), dtype=np.float64)

    abruptness = np.diff(np.diff(curve, axis = 0), axis = 0).sum(axis = 0)

    return MSE + gamma*abruptness

#prss.uses(mse)
prss.requires('curve', 'Matrix of shape (T, X1, ..., Xn) that contains the value of the predicted curve in each of T uniformly distributed points of the axis for each variable')


# TODO: Implement it with 'uses' tool
#   @evaluation_function
#   def vnprss(self, gamma):
#       """Evaluates the goodness of fit by means of the Variance Normalized Penalized Residual Sum of Squares.
#           In particular, this method computes the following expression: VNPRSS = PRSS(gamma)/VAR, that is, the Penalized Residual
#           Sum of Squares normalized with the variance of the curve.
#       """
#       try:
#           fitting_results = self.fitting_results
#       except AttributeError:
#           fitting_results = None
#   
#       PRSS = prss[self.target].evaluate(fitting_results = fitting_results, gamma = gamma)
#       curve = np.array(prss[self.target].curve(), dtype=np.float64)
#   
#       VAR = curve.var(axis = 0)
#   
#       return PRSS/VAR
#   
#   #vnprss.uses(prss)


@evaluation_function
def vnprss(self, gamma):
    """Evaluates the goodness of fit by means of the Variance Normalized Penalized Residual Sum of Squares.
        In particular, this method computes the following expression: VNPRSS = PRSS(gamma)/VAR, that is, the Penalized Residual
        Sum of Squares normalized with the variance of the curve.
    """
    try:
        fitting_results = self.fitting_results
    except AttributeError:
        fitting_results = None

    MSE = mse[self.target].evaluate(fitting_results)
    curve = np.array(self.curve(), dtype=np.float64)

    VAR = curve.var(axis = 0)

    abruptness = np.diff(np.diff(curve, axis = 0), axis = 0).sum(axis = 0)

    return (MSE + gamma*abruptness)/VAR

#vnprss.uses(mse)
vnprss.requires('curve', 'Matrix of shape (T, X1, ..., Xn) that contains the value of the predicted curve in each of T uniformly distributed points of the axis for each variable')


