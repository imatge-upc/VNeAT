#Patterns

This document explains how the fitted curves, the F-scores and the p-values of the subjects in the FPM database are computed.


##Features and GLM initialization

Firstly, we obtain the features of each subject, i.e., their **sex** (only linear term) and **age** (linear and quadratic terms), and the **AD-CSF index** associated to each of them (up to the cubic term).

```python
sex_terms = [subjects.sex] # only linear term
age_terms = polynomials(subjects.age, max_degree = 2) # terms up to degree 2
adcsf_terms = polynomials(subjects.adcsf, max_degree = 3) # terms up to degree 3
```


Then we introduce the terms associated to sex and age, together with the intercept term (all ones) in matrix **_xdata1_** and the ones associated to the AD-CSF index in **_xdata2_** and **_xdata3_**, as follows:

```python
nsubjects = len(subjects)
xdata1 = [1]*nsubjects + sex_terms + age_terms
xdata2 = linear(adcsf_terms) + nonlinear(adcsf_terms)
xdata3 = nonlinear(adcsf_terms) + linear(adcsf_terms)
```

> Note that *xdata2* and *xdata3* have the same features in different order (in particular, the linear term has been moved to the last position in *xdata3*). In this snippet it is assumed that the '+' operation concatenates lists as if they were columns, that is, the original features will be contained in the **columns** of the matrices *xdata1*, *xdata2* and *xdata3* rather than in their rows.


Finally, we create 4 instances of the GLM class, such that:
	
* The first instance contains the features to be used as **correctors** of the gray matter data, that is, the *xdata1* matrix. We will **orthonormalize** the features in this matrix to make it easier for the algorithm to compute the parameters that best fit the data for each voxel.

* The second instance contains the features to be used to compute the **coefficients** of the **polynomial curve** in the AD-CSF index, i.e., this instance will have the **non-orthonormalized** *xdata2* matrix as the model matrix. Notice that, in this case, we can't orthogonalize the columns, since we need them to preserve their meaning so that we are then able to compute the value of the fitted curve for any value of the AD-CSF index.

* The third instance contains the model matrix needed to perform F-tests over the significance of the nonlinear terms of the AD-CSF index. This is the **orthonormalized** *xdata2* matrix.

* The fourth and last instance contains the model matrix to perform F-tests over the significance of the linear term of the AD-CSF index, i.e., the **orthonormalized** *xdata3* matrix.

```python
# Correction GLM
glm1 = GLM(xdata1)
glm1.orthonormalize()

# Curve GLM
glm2 = GLM(xdata2)

# Nonlinear test GLM
glm3 = GLM(xdata2)
glm3.orthonormalize()

# Linear test GLM
glm4 = GLM(xdata3)
glm4.orthonormalize()
```


##Computation of correction, curve and test parameters

We call *correction parameters* to the parameters in the first instance of the GLM class (the one with the correction features as model matrix) that give the best fit of the model for the original gray matter (GM) values.

Similarly, we name the optimal parameters in the second instance of the GLM class *curve parameters* (these parameters will be the coefficients of the AD-CSF polynomial expression that best fits the corrected GM values).

And last, we call *test parameters* to the parameters of both the 3rd and the 4th GLM instances. These parameters will be later used to compute the F-scores and the p-values of each voxel (see chapter *Obtention of F-scores and p-values*).

The computation of all the parameters is performed inside the function *optimize* of the GLM class, by calculating the result of the expression for the optimal parameters in an *Ordinary Least Squares* problem. Thus, we only need to provide the observations for which we want to obtain the optimal parameters of the model, i.e., the **original GM values** for each voxel in the case of the first GLM, and the **corrected GM values** for the rest of the GLMs.

```python
glm1.ydata = gmvalues
glm1.optimize()

prediction = GLM.predict(glm1.xdata, glm1.opt_params)

corrected_gmvalues = gmvalues - prediction

glm2.ydata = corrected_gmvalues
glm2.optimize()

glm3.ydata = corrected_gmvalues
glm3.optimize()

glm4.ydata = corrected_gmvalues
glm4.optimize()
```


The optimal parameters for each GLM instance are contained in the *opt_params* attribute after calling the *optimize* function.


## Obtention of F-scores and p-values

For each of the two last GLM instances (*glm3* and *glm4*), we perform an F-test as follows:

1. Obtain the *restricted model*:

	* For the nonlinear test, this is a column-matrix containing only the linear term of the AD-CSF index.

	* For the linear test, it is a matrix containing both nonlinear terms of the AD-CSF index.

2. Compute the prediction of the corrected GM values using the *restricted model*.

3. Compute the error vector for the *restricted model*.

4. Compute the Residual Sum of Squares of the *restricted model* (*RSS1*).

5. Compute the prediction of the corrected GM values using the *full model*.

6. Compute the error vector for the *full model*.

7. Compute the Residual Sum of Squares of the *full model*. (*RSS2*).

8. Compute the F-score.


The F-scores in the last step are computed in accordance to the next formula:

![( (*RSS1* - *RSS2*) / (*p2* - *p1*) ) / ( *RSS2* / (*n* - *p2* + 1) )](https://upload.wikimedia.org/math/4/5/e/45edd54391e1715910f718d7807f7250.png)

> Note: Here *p1* and *p2* denote the number of parameters/regressors (number of columns) present in the *restricted model* and the *full model* respectively, whereas *n* denotes the number of samples (subjects) in the system.

```python
# Nonlinear f-score

pred1 = GLM.predict(linear(glm3.xdata), linear(glm3.opt_params)) # steps 1, 2
err1 = corrected_gmvalues - pred1 # step 3
rss1 = sum(err1**2) # step 4

pred2 = GLM.predict(glm3.xdata, glm3.opt_params) # step 5
err2 = corrected_gmvalues - pred2 # step 6
rss2 = sum(err2**2) # step 7

# step 8
p1_nl = 1 # only the linear term
p2_nl = num_regressors(glm3.xdata)
f_nonlinear = ((rss1 - rss2)/(p2_nl - p1_nl)) / (rss2/(nsubjects - p2_nl + 1))


# Linear f-score

pred1 = GLM.predict(nonlinear(glm4.xdata), nonlinear(glm4.opt_params)) # steps 1, 2
err1 = corrected_gmvalues - pred1 # step 3
rss1 = sum(err1**2) # step 4

pred2 = GLM.predict(glm4.xdata, glm4.opt_params) # step 5
err2 = corrected_gmvalues - pred2 # step 6
rss2 = sum(err2**2) # step 7

# step 8
p1_l = num_regressors(glm4.xdata) - 1 # all except the linear term
p2_l = num_regressors(glm4.xdata)
f_linear = ((rss1 - rss2)/(p2_l - p1_l)) / (rss2/(nsubjects - p2_l + 1))
```

Under the null hypothesis that the full model does not provide a significantly better fit than the restricted model, this F-score will have an F-distribution with **(*p2* − *p1*, *n* − *p2* + 1)** degrees of freedom. Hence, we can compute the **p-value** as shown below:

```python
# Nonlinear p-value
df1 = p2_nl - p1_nl # Degrees of freedom for the restricted model
df2 = nsubjects - p2_nl + 1 # Degrees of freedom for the full model
pvalue_nonlinear = 1 - f_statistic.cdf(f_nonlinear, df1, df2)

# Linear p-value
df1 = p2_l - p1_l # Degrees of freedom for the restricted model
df2 = nsubjects - p2_l + 1 # Degrees of freedom for the full model
pvalue_linear = 1 - f_statistic.cdf(f_linear, df1, df2)
```
> Note: *f_statistic.cdf* is a function that computes the Cumulative Density Function of an F-distributed variable with the provided degrees of freedom at the point specified by the first parameter.


## Filtering of invalid voxels and visualization of the F-scores

Now that we have the f-scores (and the corresponding p-values) for both the nonlinear and the linear components of the model for each voxel, we will filter all the voxels that don't meet the following requirements:

* Having a large enough (> 0.2) mean amount of gray matter volume (meaning they don't belong to noisy zones in the brain's image).

* Having a small enough (< 0.001) p-value in either of the components of the model (meaning they have a strong dependence on at least one of the linear or the nonlinear terms of the AD-CSF index).

* Belonging to a big enough cluster (>= 100 elements) of contiguous voxels that fulfill the previous conditions (to avoid false positives).

To this end, we first compute the minimum p-value between the one corresponding to the linear term and the one corresponding to the nonlinear terms for each voxel.

```python
min_pvalue = min(pvalue_nonlinear, pvalue_linear)
```

Next, we compute the mean value of GM volume for each voxel, and test whether it exceeds the required minimum or not. If not, the corresponding *minimum p-value* is set to 1.0 so that this voxel is filtered afterwards:

```python
if mean(gmvalues) < gm_threshold:
	min_pvalue = 1.0
```

Finally, all *minimum p-values* are stored in a 3D matrix called *min_pvalues*, which is then treated as an undirected graph such that each voxel is a node and an edge *(u -> v)* exists if and only if voxels *u* and *v* are contiguous and both have a *minimum p-value* of **at most _pv_threshold_**. By applying a *Strongly Connected Components* algorithm to such graph, we obtain the clusters of the image, being thus able to test whether their size is above the required level (*minimum_num_nodes_cluster*).

```python
g = Graph(min_pvalues, pv_threshold)
for scc in g.SCCs():
	if size(scc) < minimum_num_nodes_cluster:
		for each voxel in scc:
			voxel.fscore_linear = 0
			voxel.fscore_nonlinear = 0
```

> Note: Since this graph is undirected, a simple *Breadth First Search* algorithm can easily compute the Strongly Connected Components of the brain image.

For visualization purposes, we also re-scale the f-scores so that they are in the range *[0, 1]*, by dividing them by the maximum value in the whole image. The resulting scores for the linear term are interpreted as the red component of a colormap, whereas the scores for the nonlinear term are taken as the green component of such map, in a way that when visualizing a voxel, we will be able to show a color that indicates the level of nonlinearity vs. the level of linearity that the GM values of that voxel have with respect to the AD-CSF index:

```python
max_fscore = fscores.max()
fscores /= max_fscore
```


## Computation of curves

As a result of having computed the optimal parameters of the **non-orthogonalized** model that contains the polynomial terms of the AD-CSF index, we can now obtain a certain voxel's fitted curve's value for any desired value of the AD-CSF index by just taking the latter number and its second and third powers, and multiplying them by the parameters obtained for the voxel of interest.

This is the same as predicting the corrected GM volume of a voxel given a certain value for the AD-CSF index by applying a GLM with the linear, quadratic and cubic powers of such value as the model and the previously obtained parameters as the optimal coefficients of the model.

Therefore, given a range *[a, b]* and a number of points *nPoints* in which to evaluate the curve's value, the curve (for a certain voxel) can be displayed as follows:

```python
adcsf = linspace(a, b, nPoints) # create 'nPoints' points uniformly distributed between a and b
adcsf_terms = polynomials(adcsf, max_degree = 3) # get powers 1, 2 and 3 of the points
curve = GLM.predict(adcsf_terms, glm2.opt_params)
plot(curve)
```



