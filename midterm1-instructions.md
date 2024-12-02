[![Repo Checks](https://github.com/Sta663-Sp22/midterm1_eschmidt45/workflows/Repo%20Checks/badge.svg)](https://github.com/Sta663-Sp22/midterm1_eschmidt45/actions?query=workflow:%22Repo%20Checks%22)


## Sta 663 - Statistical Computing and Computation - Midterm 1

Due Monday, February 28th by 5:00 pm.

## Rules

1.  Your solutions must be written up using the provided iPython
    notebook file (`midterm1.ipynb)`, this file must include your code
    and write up for each task.

2.  This project is open book, open internet, closed other people. You
    may use *any* online or book based resource you would like, but you
    must include citations for any code that you use (directly or
    indirectly). You *may not* consult with anyone else about this exam
    other than the Lecturers or Tutors for this course - this includes
    posting anything online. You may post questions on Piazza, general
    questions are fine and can be posted publicly - any question
    containing code should be posted privately.

3.  If you receive help *or* provide help to any other student in this
    course you will receive a grade of 0 for this assignment. Do not
    share your code with anyone else in this course.

4.  You have until Monday, February 28th by 5:00 pm to complete this
    assignment and turn it in via your personal Github repo - late work
    will be subject to the standard late penalty. Technical difficulties
    are not an excuse for late work - do not wait until the last minute
    to commit / push.

5.  All of your answers *must* include a brief description / write up of
    your approach. This includes both annotating / commenting your code
    *and* a separate written descriptions of all code / implementations.
    I should be able to suppress *all* code output in your document and
    still be able to read and make sense of your answers.

6.  You may use any packages discusses in class this far (numpy, scipy,
    matplotlib, pandas, seaborn), if there are any additional packages
    you would like to use you must clear them with me first.

7.  Your first goal is to write code that can accomplish all of the
    given tasks, however keep in mind that marking will also be based on
    the quality of the code you write - elegant, efficient code will be
    given better marks and messy, slow code will be penalized.

<br />

------------------------------------------------------------------------

## Gaussian Process Regression

For this assignment you will be implementing a number of functions which
will be used to fit a non-linear function to 1-dimensional data using
Gaussian process regression. We do not assume that you are previously
familiar with this method and will provide all necessary details to
implement the models and related algorithms - some familiarity with the
basic distribution theory for multivariate normal distributions will be
helpful.

### Data

The `data/` directory contains three sample data sets (`d1.csv`,
`d2.csv`, and `d3.csv`) which reflect noisy measurements of non-linear
smooth functions in 1d. Each data set is composed of different numbers
of observations of `x` and `y` pairs - it is highly recommended that you
create scatter plots of all three to get a sense of the relationships
present and the ranges of the values.

Each data set has been read into your environment as pandas data frames
`d1`, `d2`, and `d3` respectively.

------------------------------------------------------------------------

### Task 1 - Fitting

For this task your goal is to write a generic function which will be
able to fit a Gaussian process regression model to each of provided data
sets via maximum likelihood estimation.

#### Background

For a Gaussian process regression we will assume a model with the
following form,

![y \\sim \\text{MVN}(\\mu, \\Sigma(x))](https://latex.codecogs.com/svg.latex?y%20%5Csim%20%5Ctext%7BMVN%7D%28%5Cmu%2C%20%5CSigma%28x%29%29 "y \sim \text{MVN}(\mu, \Sigma(x))")

which implies that our observed values of
![y](https://latex.codecogs.com/svg.latex?y "y") are derived from a
multivariate normal distribution with a specific mean and covariance
structure. For the sake of simplicity, we will assume that
![\\mu](https://latex.codecogs.com/svg.latex?%5Cmu "\mu") will always be
0 (a vector of zeros in this case) for these models. The covariance
(![\\Sigma](https://latex.codecogs.com/svg.latex?%5CSigma "\Sigma"))
will be given by a Gaussian / squared exponential kernel that is a
function of the ![x](https://latex.codecogs.com/svg.latex?x "x") values,
specifically their distances from one another.

Explicitly, the elements of the covariance matrix are constructed using
the following formula,

![ \\Sigma\_{i,j}(x) = \\text{cov}(x_i, x_j) = \\sigma^2_n \\, \\mathcal{I}\_{i = j} + \\sigma^2_s \\exp{\\left(- \\frac{(x_i-x_j)^2}{2l}\\right)} ](https://latex.codecogs.com/svg.latex?%20%5CSigma_%7Bi%2Cj%7D%28x%29%20%3D%20%5Ctext%7Bcov%7D%28x_i%2C%20x_j%29%20%3D%20%5Csigma%5E2_n%20%5C%2C%20%5Cmathcal%7BI%7D_%7Bi%20%3D%20j%7D%20%2B%20%5Csigma%5E2_s%20%5Cexp%7B%5Cleft%28-%20%5Cfrac%7B%28x_i-x_j%29%5E2%7D%7B2l%7D%5Cright%29%7D%20 " \Sigma_{i,j}(x) = \text{cov}(x_i, x_j) = \sigma^2_n \, \mathcal{I}_{i = j} + \sigma^2_s \exp{\left(- \frac{(x_i-x_j)^2}{2l}\right)} ")

where
![\\sigma^2_n](https://latex.codecogs.com/svg.latex?%5Csigma%5E2_n "\sigma^2_n"),
![\\sigma^2_s](https://latex.codecogs.com/svg.latex?%5Csigma%5E2_s "\sigma^2_s")
and ![l](https://latex.codecogs.com/svg.latex?l "l") are the models
parameters (what we will be estimating via MLE from the data).

-   ![\\sigma^2_n](https://latex.codecogs.com/svg.latex?%5Csigma%5E2_n "\sigma^2_n") -
    this is the nugget variance parameter and represents irreducible
    measurement error. Note:
    ![\\mathcal{I}\_{i = j}](https://latex.codecogs.com/svg.latex?%5Cmathcal%7BI%7D_%7Bi%20%3D%20j%7D "\mathcal{I}_{i = j}")
    is an indicator function which is 1 when
    ![i=j](https://latex.codecogs.com/svg.latex?i%3Dj "i=j") and 0
    otherwise, meaning the nugget variance is only added to the diagonal
    of covariance matrix.
-   ![\\sigma^2_s](https://latex.codecogs.com/svg.latex?%5Csigma%5E2_s "\sigma^2_s") -
    this is the scale variance parameter and determines the average
    distance away from the mean that can be taken by the function.
-   ![l](https://latex.codecogs.com/svg.latex?l "l") - this is the
    length-scale parameter which determines the range of the “spatial”
    dependence between points. Larger values of
    ![l](https://latex.codecogs.com/svg.latex?l "l") result in *less*
    wiggly functions (greater spatial dependence) and smaller values
    result in *more* wiggly functions (lesser spatial dependence) -
    values are relative to the scale of
    ![x](https://latex.codecogs.com/svg.latex?x "x").

#### Model fitting process

In order to fit the model the goal is to determine the optimal values of
these three parameters given the data. We will be accomplishing this via
maximum likelihood. Given our multivariate normal model we can then take
the MVN density,

![
f(y) = \\frac{1}{\\sqrt{\\det(2\\pi\\Sigma(x))}} \\exp \\left\[-\\frac{1}{2} (y-\\mu)^T \\Sigma(x)^{-1} (y-\\mu) \\right\]
](https://latex.codecogs.com/svg.latex?%0Af%28y%29%20%3D%20%5Cfrac%7B1%7D%7B%5Csqrt%7B%5Cdet%282%5Cpi%5CSigma%28x%29%29%7D%7D%20%5Cexp%20%5Cleft%5B-%5Cfrac%7B1%7D%7B2%7D%20%28y-%5Cmu%29%5ET%20%5CSigma%28x%29%5E%7B-1%7D%20%28y-%5Cmu%29%20%5Cright%5D%0A "
f(y) = \frac{1}{\sqrt{\det(2\pi\Sigma(x))}} \exp \left[-\frac{1}{2} (y-\mu)^T \Sigma(x)^{-1} (y-\mu) \right]
")

we can derive the log likelihood as

![ 
\\ln L(y) = -\\frac{1}{2} \\left\[n \\ln (2\\pi) + \\ln (\\det \\Sigma(x)) + (y-\\mu)^T \\Sigma(x)^{-1} (y-\\mu) \\right\].
](https://latex.codecogs.com/svg.latex?%20%0A%5Cln%20L%28y%29%20%3D%20-%5Cfrac%7B1%7D%7B2%7D%20%5Cleft%5Bn%20%5Cln%20%282%5Cpi%29%20%2B%20%5Cln%20%28%5Cdet%20%5CSigma%28x%29%29%20%2B%20%28y-%5Cmu%29%5ET%20%5CSigma%28x%29%5E%7B-1%7D%20%28y-%5Cmu%29%20%5Cright%5D.%0A " 
\ln L(y) = -\frac{1}{2} \left[n \ln (2\pi) + \ln (\det \Sigma(x)) + (y-\mu)^T \Sigma(x)^{-1} (y-\mu) \right].
")

The goal therefore is to find,

![\\underset{\\sigma^2_n, \\sigma^2_s, l}{\\text{argmax}} \\, L(y) \\quad \\text{or} \\quad \\underset{\\sigma^2_n, \\sigma^2_s, l}{\\text{argmin}} \\, -L(y).](https://latex.codecogs.com/svg.latex?%5Cunderset%7B%5Csigma%5E2_n%2C%20%5Csigma%5E2_s%2C%20l%7D%7B%5Ctext%7Bargmax%7D%7D%20%5C%2C%20L%28y%29%20%5Cquad%20%5Ctext%7Bor%7D%20%5Cquad%20%5Cunderset%7B%5Csigma%5E2_n%2C%20%5Csigma%5E2_s%2C%20l%7D%7B%5Ctext%7Bargmin%7D%7D%20%5C%2C%20-L%28y%29. "\underset{\sigma^2_n, \sigma^2_s, l}{\text{argmax}} \, L(y) \quad \text{or} \quad \underset{\sigma^2_n, \sigma^2_s, l}{\text{argmin}} \, -L(y).")

#### Details

For this task you will need to implement a python function which
implements maximum likelihood estimation for this class of model and
meets the following requirements:

-   The maximum likelihood estimate should be made using an appropriate
    optimizer

    -   In order to make the process simpler - you may treat
        ![\\sigma^2_n](https://latex.codecogs.com/svg.latex?%5Csigma%5E2_n "\sigma^2_n")
        as fixed with a value of 0.1 (which is the true value for all
        three data sets).
    -   There is no simple closed form for the gradient of the
        likelihood in terms of
        ![l](https://latex.codecogs.com/svg.latex?l "l") and
        ![\\sigma^2_s](https://latex.codecogs.com/svg.latex?%5Csigma%5E2_s "\sigma^2_s"),
        so choose an optimization method that does not require the
        gradient.
    -   All parameters must be
        ![\\geq 0](https://latex.codecogs.com/svg.latex?%5Cgeq%200 "\geq 0")
        to produce a valid covariance, construct an appropriate
        `Bounds()` object to aid in the optimization.
    -   Optimization methods may fail based on the initial guess
        provided - as such you should initialize at multiple starting
        points and only keep results which succeed
        -   If all optimization attempts fail then raise a
            `RuntimeError`.
        -   Feel free to ignore any warnings that are produced by the
            optimizer (so long as at least one attempt succeeds)

-   Write a function named `fit()` which takes the following arguments:

    -   `d` - data set to fit (pandas data frame with `x` and `y`
        columns)
    -   `n_init` - number of initial guesses to use (i.e. number of
        optimizations to run)
    -   `lb` - list (or array) of parameter lower bounds - we suggest
        not using 0 but rather a small value like `1e-6`.
    -   `ub` - list (or array) of parameter upper bounds - we suggest
        `[0.1, 5, 2]` as reasonable defaults.
    -   `seed` - a seed value which will be used to initialize
        `numpy.random.default_rng()`, which will then be used for all
        random number generation within the function.

-   Your function must return a list with the MLEs of
    ![\\sigma^2_n](https://latex.codecogs.com/svg.latex?%5Csigma%5E2_n "\sigma^2_n"),
    ![\\sigma^2_s](https://latex.codecogs.com/svg.latex?%5Csigma%5E2_s "\sigma^2_s"),
    ![l](https://latex.codecogs.com/svg.latex?l "l") (3 double values)

    -   If more than one of the optimization attempts succeeded, select
        the best and return only those MLE values.

-   Make sure that all example calls to `fit()` do not use the same
    seed.

-   Include a brief write up describing your function and implementation
    approach.

------------------------------------------------------------------------

### Task 2 - Prediction

#### Background

Once the model parameters have been obtained the goal will be to predict
(i.e. draw samples from) our Gaussian process model for new values of
![x](https://latex.codecogs.com/svg.latex?x "x"). Specifically, we want
to provide an fine, equally space grid of
![x](https://latex.codecogs.com/svg.latex?x "x") values from which we
will predict the value of the function. Multiple independent predictions
(draws) can then be average to get an overall estimate of the underlying
smooth function for each data set.

Therefore the goal is to find the conditional predictive distribution of
![y_p](https://latex.codecogs.com/svg.latex?y_p "y_p") given
![y](https://latex.codecogs.com/svg.latex?y "y"),
![x](https://latex.codecogs.com/svg.latex?x "x"),
![x_p](https://latex.codecogs.com/svg.latex?x_p "x_p"), and
![\\theta = (\\sigma^2_n, \\sigma^2_s, l)](https://latex.codecogs.com/svg.latex?%5Ctheta%20%3D%20%28%5Csigma%5E2_n%2C%20%5Csigma%5E2_s%2C%20l%29 "\theta = (\sigma^2_n, \sigma^2_s, l)").
Given everything is a multivariate normal distribution, this conditional
distribution is

![ y_p \| y, \\theta \\sim \\text{MVN}(\\mu^\\star, \\Sigma^\\star)](https://latex.codecogs.com/svg.latex?%20y_p%20%7C%20y%2C%20%5Ctheta%20%5Csim%20%5Ctext%7BMVN%7D%28%5Cmu%5E%5Cstar%2C%20%5CSigma%5E%5Cstar%29 " y_p | y, \theta \sim \text{MVN}(\mu^\star, \Sigma^\star)")

  
where

![
\\begin{align\*}
\\mu^\\star &= \\mu_p + \\Sigma(x_p, x) \\, \\Sigma(x)^{-1} \\, (y - \\mu) \\\\
\\Sigma^\\star &= \\Sigma(x_p) - \\Sigma(x_p, x) \\, \\Sigma(x)^{-1} \\, \\Sigma(x, x_p)
\\end{align\*}
](https://latex.codecogs.com/svg.latex?%0A%5Cbegin%7Balign%2A%7D%0A%5Cmu%5E%5Cstar%20%26%3D%20%5Cmu_p%20%2B%20%5CSigma%28x_p%2C%20x%29%20%5C%2C%20%5CSigma%28x%29%5E%7B-1%7D%20%5C%2C%20%28y%20-%20%5Cmu%29%20%5C%5C%0A%5CSigma%5E%5Cstar%20%26%3D%20%5CSigma%28x_p%29%20-%20%5CSigma%28x_p%2C%20x%29%20%5C%2C%20%5CSigma%28x%29%5E%7B-1%7D%20%5C%2C%20%5CSigma%28x%2C%20x_p%29%0A%5Cend%7Balign%2A%7D%0A "
\begin{align*}
\mu^\star &= \mu_p + \Sigma(x_p, x) \, \Sigma(x)^{-1} \, (y - \mu) \\
\Sigma^\star &= \Sigma(x_p) - \Sigma(x_p, x) \, \Sigma(x)^{-1} \, \Sigma(x, x_p)
\end{align*}
")

In these formulae,
![\\Sigma(x_p)](https://latex.codecogs.com/svg.latex?%5CSigma%28x_p%29 "\Sigma(x_p)")
is the
![n_p \\times n_p](https://latex.codecogs.com/svg.latex?n_p%20%5Ctimes%20n_p "n_p \times n_p")
covariance matrix constructed from the
![n_p](https://latex.codecogs.com/svg.latex?n_p "n_p") prediction
locations and
![\\Sigma(x_p, x)](https://latex.codecogs.com/svg.latex?%5CSigma%28x_p%2C%20x%29 "\Sigma(x_p, x)")
is the
![n_p \\times n](https://latex.codecogs.com/svg.latex?n_p%20%5Ctimes%20n "n_p \times n")
cross covariance matrix constructed from the
![n_p](https://latex.codecogs.com/svg.latex?n_p "n_p") prediction
locations and the ![n](https://latex.codecogs.com/svg.latex?n "n") data
locations. Note that
![\\Sigma(x_p, x)^T = \\Sigma(x, x_p)](https://latex.codecogs.com/svg.latex?%5CSigma%28x_p%2C%20x%29%5ET%20%3D%20%5CSigma%28x%2C%20x_p%29 "\Sigma(x_p, x)^T = \Sigma(x, x_p)").
As mentioned in the preceding task - we will assume that
![\\mu](https://latex.codecogs.com/svg.latex?%5Cmu "\mu") and
![\\mu_p](https://latex.codecogs.com/svg.latex?%5Cmu_p "\mu_p") are 0.

#### Details

For this task you will need to implement a python function which
calculates the mean and covariance of conditional distribution described
above and meets the following requirements:

-   Write a function named `predict()` which takes the following
    arguments:

    -   `d` - data set that was fit (must match the data frame used to
        calculate theta)
    -   `theta` - list of MLEs of
        ![\\sigma^2_n](https://latex.codecogs.com/svg.latex?%5Csigma%5E2_n "\sigma^2_n"),
        ![\\sigma^2_s](https://latex.codecogs.com/svg.latex?%5Csigma%5E2_s "\sigma^2_s"),
        and ![l](https://latex.codecogs.com/svg.latex?l "l") from \`
    -   `x_p` - array of `x` prediction locations (should span the range
        of `x`s in `d`)
    -   `n_draws` - number of draws to make from the conditional
        distribution
    -   `seed` - a seed value which will be used to initialize
        `numpy.random.default_rng()` for this function.

-   Your function must return a pandas data frame with 1+`n_draws`
    columns: one `x` column (`x_p`) and `n_draws` `y` columns,

    -   Draws can be made via the `multivariate_normal()` method of your
        `numpy.random.default_rng()` instance or via a cholesky
        decomposition with unit normal draws.
    -   Generate samples as efficiently as possible (avoid repeated
        calls to `multivariate_normal()`)
    -   DataFrame should be returned in a “wide” format, all columns
        should have a reasonable name.

-   Make sure that all example calls to `predict()` do not use the same
    seed.

-   Include a brief write up describing your function and implementation
    approach.

------------------------------------------------------------------------

### Task 3 - Plotting

For this task you will need to take the result from the `predict()`
function and generate a plot showing the mean predicted `y` (across
prediction samples) as well as a shaded region showing a 95% confidence
interval (empirically determined from the prediction samples).

Optionally, the user should be able to provide the original data set `d`
which would then be overlayed as a scatter plot.

-   Write a function named `plot_gp()` which takes the following
    arguments:
    -   `pred` - data frame of predictions from the `predict()` function
    -   `d` - either `None` or a data frame with data observations
-   Your function does not need to return anything, but should display a
    pyplot figure with all of the above features.
    -   Feel free to hard code things like `figsize` so that the
        resulting figures look reasonable in your notebook.
-   Include a brief write up describing your function and implementation
    approach.
