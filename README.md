# UmbrellaIntegrate.py
Umbrella Integration algorithm of calculating PMF using Python

## Dependence

`Python3`
`Numpy`
`pandas` for parsing metafile
`Scipy` for `simps` integration


## Usage:

```bash
python3 ubint.py <your-metafile>
```

### Input

#### Metafile

The `<your-metafile>` should be in fellowing form:

```bash
/path/to/your/window/file window_center spring constant [temperature]
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
