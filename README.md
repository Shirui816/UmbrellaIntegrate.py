# UmbrellaIntegrate.py
Umbrella Integration<sup>[1]</sup> algorithm of calculating PMF using Python.

## Dependence

* `Python3`
* `Numpy`
* `pandas` for parsing metafile
* `Scipy` for integration


## Usage:

See help:
```bash
python3 ubint.py -h
```

### Input

#### Metafile

The `<your-metafile>` should be in fellowing form:

```bash
/path/to/your/window/file window_center spring_constant [temperature]
```

There is a variable of `T` in `ubint.py`, if the `temperature` left blank in the metafile, the default temperature would be variable `T` in the `ubint.py`, or you can set specific temperature for some window.

#### Data file for each window

The data file of each window need to be a 2-column file with `time reaction_coordinate`, the coordinate should be 1-dimensional.

### Output

The output file is `free_py.txt` with 2-column `reaction_coordinate free_energy`

## Warning

### Unit

I use `kJ/mol` in this program.

### Spring constant `K`

In your simulation, the biased spring potential shoud be in form of `0.5 * K * (r - r0) ** 2`, here `K` is the parameter set in your `<your-metafile>`, for some simulation program, there is no `0.5` in the biased spring potential.

## Screen shots

Raw data was generated by [Gaussian distribution](https://en.wikipedia.org/wiki/Normal_distribution) for each window with `MEAN=window_center` and `STD=0.8`, the centers are in range of `0.0 ~ 19.5` by step of `0.5`, here is the result compare with WHAM<sup>[2]</sup>:

* Raw Data

![Raw Raw](https://raw.githubusercontent.com/Shirui816/UmbrellaIntegrate.py/master/ScreenShot/DataDetail.png)

![Raw IL](https://raw.githubusercontent.com/Shirui816/UmbrellaIntegrate.py/master/ScreenShot/Data.png)

* Compare with WHAM

![CMP CMP](https://raw.githubusercontent.com/Shirui816/UmbrellaIntegrate.py/master/ScreenShot/PMF_UI_WHAM.png)

**The zero point in WHAM is the minimum value and the zero point in UI is 0.**

## TO DO

The UI algorithm with higher oder terms<sup>[3]</sup> of `A(xi)` is `ubint_ho_devel.py`, the result is not ideal using previous data, still in development.

**Problems occurred at standard normal distributions, maybe the quadruplicate term which even possesses a small value could cause a huge deviation. I should try some systems with non-quadratic potentials.**

**The function `exp(-beta(a1*xi+a2*xi^2+a3*xi^3+a4*xi^4))` and its integration (Normalization factor) give very large value (even inf), this is unable to solve yet.**

**Solved**
Use KDE method to evaluate `a_i` via `curve_fit` with an appropriate range, guarantee the convergence of mean force in the whole range. The results are fine.

## Ref

1. Kästner, Johannes, and Walter Thiel. “Bridging the Gap between Thermodynamic Integration and Umbrella Sampling Provides a Novel Analysis Method: ‘Umbrella Integration.’” The Journal of Chemical Physics 123, no. 14 (October 8, 2005): 144104. doi:10.1063/1.2052648.
2. http://membrane.urmc.rochester.edu/content/wham
3. Kästner, Johannes. “Umbrella Integration with Higher-Order Correction Terms.” The Journal of Chemical Physics 136, no. 23 (June 21, 2012): 234102. doi:10.1063/1.4729373.
