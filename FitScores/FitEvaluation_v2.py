import types

import numpy as np
from scipy.stats import f as f_stat

class UNBOUND(object):
    pass

class dictlike(type):
    class _bound_generic_evaluation_function(object):
        def __init__(self, target, parent):
            self._target = target
            self._parent = parent
            self._set = []

        @property
        def target(self):
            return self._target
        
        def bind(self, method_name, method):
            setattr(self, method_name, method)
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

        def __getitem__(self, target):
            return self._parent[target]


    def __init__(self, *args, **kwargs):
        super(dictlike, self).__init__(*args, **kwargs)
        self._bindings = {}

    def __getitem__(self, target):
        try:
            return self._bindings[target]
        except KeyError:
            retval = dictlike._bound_generic_evaluation_function(target, self)
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
        def __init__(self, parent, target):
            self._parent = parent
            self._target = target
            self._evaluate = self._parent._evaluate
            self._requirements = [evaluation_function.RequirementDescriptor(name, desc, UNBOUND) for (name, desc) in self._parent._requirements.iteritems()]
            self._implicit = map(lambda rd: evaluation_function.RequirementDescriptor(rd.name, rd.description, types.MethodType(rd.value, self)), self._parent._implicit.itervalues())
            self._dependencies = self._parent._dependencies

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

        def _frost_and_inherit(self, status=False):
            try:
                mro = self._target.mro()[1:] # the target is analyzed separately
            except AttributeError:
                try:
                    mro = type(self._target).mro()
                except AttributeError:
                    mro = [] # [self._target] if isinstance(self._target, type) else [type(self._target)]


            frosted = []
            inherited = []

            # Freeze all dependencies so that their bindings are accessible as attributes
            if not status:
                for (alias, eval_func) in self._dependencies.iteritems():
                    try:
                        # Copy the binding of the target and the evaluation function on which the current one depends,
                        # freeze such copy and set it as an attribute of this binding
                        setattr(self, alias, eval_func[self._target]._copy())
                        try:
                            setattr(getattr(self, alias), 'fitting_results', self.fitting_results)
                        except AttributeError:
                            pass
                        getattr(self, alias).freeze()
                        frosted.append(alias)
                    except Exception as e:
                        self._revert_inheritance_and_defrost(inherited, frosted)
                        raise e

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
                        setattr(self, ReqDescriptor.name, types.MethodType(inherited_method, self))
                        inherited.append(ReqDescriptor.name)
                        continue
                    except (KeyError, AttributeError):
                        pass

                    # First step failed; go to second step:
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
                                inherited_method = inherited_method.im_func # Get the original, unbound method
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
                        setattr(self, ReqDescriptor.name, types.MethodType(inherited_method, self)) # Bind it to this instance
                        inherited.append(ReqDescriptor.name)
                        break
                    else: # bad news... :(
                        if not status: # ...unless we are checking the status, in which case there's no problem
                            # rollback
                            self._revert_inheritance_and_defrost(inherited, frosted)

                            # and raise
                            raise RuntimeError("Method '" + ReqDescriptor.name + "' could not be inherited.")

            # everything went smoothly
            return inherited, frosted

        def _copy(self):
            retval = evaluation_function._bound_evaluation_function(parent = self._parent, target = self._target)

            for rd in (self._requirements + self._implicit):
                own_method = getattr(self, rd.name)
                if not (own_method is UNBOUND):
                    retval.bind(rd.name, own_method.im_func)

            retval._frozen = [x for x in self._frozen]

            for method_name in self._forced:
                retval.bind(method_name, getattr(self, method_name).im_func, force=True)

            return retval

        def _revert_inheritance_and_defrost(self, inherited, frosted):
            for name in inherited:
                setattr(self, name, UNBOUND)
            for alias in frosted:
                exec('del self.' + alias)

            return self

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
            for ReqDescriptor in (self._implicit + self._requirements):
                setattr(self, ReqDescriptor.name, ReqDescriptor.value)

            self._frozen = []

            for name in self._forced:
                exec('del self.' + name)

            self._forced = []
            self._frosted = []

            return self

        def evaluate(self, fitting_results = None, *args, **kwargs):
            if not (fitting_results is None):
                try:
                    previous_fitting_results = self.fitting_results
                except AttributeError:
                    previous_fitting_results = None

                self.fitting_results = fitting_results

            inherited, frosted = self._frost_and_inherit()

            # all clear, do the job!
            try:
                retval = self._evaluate(self, *args, **kwargs)
            except Exception as e:
                # oops! something went wrong; rollback and re-raise
                self._revert_inheritance_and_defrost(inherited, frosted)
                raise e

            # everything fine, undo inheritance and return the result
            self._revert_inheritance_and_defrost(inherited, frosted)

            if not (fitting_results is None):
                if not (previous_fitting_results is None):
                    self.fitting_results = previous_fitting_results
                else:
                    del self.fitting_results

            return retval

        def freeze(self):
            """Stops dynamic inheritance; takes a snapshot of the current status and stores it.
            """
            inherited, frosted = self._frost_and_inherit()
            self._frozen.extend(inherited)
            self._frosted.extend(frosted)

            return self

        def unfreeze(self):
            """Restarts dynamic inheritance.
            """
            self._revert_inheritance_and_defrost(self._frozen, self._frosted)
            self._frozen = []
            self._frosted = []

            return self

        def is_frozen(self):
            return not self._frozen # same as: "not (self._frozen is empty)"

        def status(self):
            desc = {}
            for rd in self._implicit:
                desc[rd.name] = '      [Default]  ' + rd.name + ': ' + rd.description
            for rd in self._requirements:
                desc[rd.name] = rd.name + ': ' + rd.description

            reqs = set(map(lambda r: r.name, self._requirements))

            inherited, frosted = self._frost_and_inherit(status=True)
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

            self._revert_inheritance_and_defrost(inherited, frosted)

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
            s += 'Dependencies:\n'
            for (alias, eval_func) in self._dependencies.iteritems():
                salias = str(alias)
                s += ' '*(13 - len(salias)) + '[' + salias + ']  ' + ('None\n' if eval_func._evaluate.__doc__ is None else eval_func._evaluate.__doc__) + '\n'
            s += '\n'

            return s

        def __getitem__(self, target):
            return self._parent[target]


    def __init__(self, func):
        self._bindings = {}
        self._evaluate = func
        self._requirements = {}
        self._implicit = {}
        self._dependencies = {}

        self.clear = self._clear

    def _check_integrity(self, method_name):
        if len(self._bindings) > 0:
            raise RuntimeError('This function has already been bound at least once; it is not possible to specify more requirements nor dependencies at this point.')
        
        if method_name in self._requirements:
            raise ValueError('The method ' + str(method_name) + ' was already set as a requirement for this evaluation function.')
        if method_name in self._implicit:
            raise ValueError('The method ' + str(method_name) + ' was already set with an implicit value for this evaluation function.')
        if method_name in self._dependencies:
            raise ValueError('The method ' + str(method_name) + ' was already set as an alias for a dependency of this evaluation function.')

    def requires(self, method_name, description):
        """Specifies that a method whose name is contained in 'method_name' argument is necessary to evaluate this test.
            The description of such method is given in 'description' argument.
        """
        self._check_integrity(method_name)
        self._requirements[method_name] = description
        return self

    def implicit(self, method_name, description, bound_method):
        """Specifies the name, description and value of a method that is required by the test to be evaluated.
            In this case, any target will call the method contained in 'bound_method' argument unless it has been
            're-bound' to another method for that specific target.

            IMPORTANT: methods bound this way take preference over inheritance.
        """
        self._check_integrity(method_name)
        self._implicit[method_name] = evaluation_function.RequirementDescriptor(method_name, description, bound_method)
        return self

    def uses(self, evaluation_method, alias):
        """Specifies the dependence of this evaluation function on the evaluation function passed in argument
            'evaluation_method'. The methods and bindings of the latter will be accessible when evaluating the
            former through the attribute called as specified in argument 'alias'.
            --------------------
            Example:
                # Initialize an arbitrary class A, and two evaluation methods em1 and em2
                # (...)

                # Define a requirement for em1
                em1.requires('three', 'A method that returns the numerical value 3')

                # Specify that em2 depends on em1 and/or its requirements and create an alias so that we can access
                # this element afterwards
                em2.uses(em1, 'em1_binding')

                # Define a requirement for em2
                em2.requires('f', 'A method that returns 3 + the evaluation of em1 on the target') 
                
                # Bind the required method to em1 so that it can evaluate object, and by inheritance, also A
                em1[object].bind('three', lambda self: 3)

                # We can now use the bindings of em1 through the 'em1_binding' alias that we set when calling the
                # 'uses' method
                em2[A].bind('f', lambda self: self.em1_binding.three() + self.em1_binding.evaluate())
                # Notice that we can even access the 'three' method, which is inherited, or evaluate 'em1'
                # itself on the current target (A)
                
                # Calling the evaluate method on em2[A] would now inherit the {object <-> em1 : three} binding and
                # set it in em2[A].em1_binding.three, to then use it in the 'f' function whenever this is called inside
                # the em2[A].evaluate method
                print em2[A].evaluate()
        """
        self._check_integrity(alias)

        # Look for cycles closed by this new dependency
        indirect_dependencies = [evaluation_method]
        Next = 0 # BFS
        # Next = -1 # DFS
        while indirect_dependencies:
            d = indirect_dependencies[Next]
            del indirect_dependencies[Next]
            if d is self:
                raise ValueError("Operation introduces cycle in dependency tree.")
            indirect_dependencies.extend(d._dependencies.values())


        self._dependencies[alias] = evaluation_method
        return self

    def __getitem__(self, target):
        try:
            return self._bindings[target]
        except KeyError:
            retval = evaluation_function._bound_evaluation_function(parent = self, target = target)
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
#   # TODO: make expression below (commented) work with all inheritance and stuff (be able to use the bindings of fstat in ftest)
#   ftest.uses(fstat)
#   # Another option would be to use something like this, but this does not resolve the issue either:
#   # ftest.implicit('fstat', "Result of evaluating the 'fstat' test on the target", lambda self: fstat[self.target].evaluate(getattr(self, 'fitting_results', None)))

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




##### Curve based measures #####


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

    MSE = self.mse.evaluate(fitting_results = fitting_results)
    curve = np.array(self.curve(), dtype=np.float64)
    dx = self.xdiff()

    diff1 = np.diff(curve, axis = 0)/dx
    diff2 = np.diff(diff1, axis = 0)/dx
    abruptness = (diff2**2).sum(axis = 0)

    return MSE + gamma*abruptness

prss.requires('curve', 'Matrix of shape (T, X1, ..., Xn) that contains the value of the predicted curve in each of T uniformly distributed points of the axis for each variable.')
prss.requires('xdiff', 'Float indicating the separation between any two contiguous points of the axis.')
# prss.implicit('mse', "Result of evaluating the 'mse' test on the target", lambda self: mse[self.target].evaluate(getattr(self, 'fitting_results', None)))
prss.uses(mse, 'mse')

# TODO: Test this
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

    PRSS = self.prss.evaluate(fitting_results = fitting_results, gamma = gamma)
    curve = np.array(self.prss.curve(), dtype=np.float64)

    VAR = curve.var(axis = 0)

    return PRSS/VAR

vnprss.uses(prss, 'prss')


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
#       MSE = self.mse()
#       curve = np.array(self.curve(), dtype=np.float64)
#   
#       VAR = curve.var(axis = 0)
#   
#       abruptness = np.diff(np.diff(curve, axis = 0), axis = 0).sum(axis = 0)
#   
#       return (MSE + gamma*abruptness)/VAR
#   
#   vnprss.requires('curve', 'Matrix of shape (T, X1, ..., Xn) that contains the value of the predicted curve in each of T uniformly distributed points of the axis for each variable')
#   vnprss.implicit('mse', "Result of evaluating the 'mse' test on the target", lambda self: mse[self.target].evaluate(getattr(self, 'fitting_results', None)))


