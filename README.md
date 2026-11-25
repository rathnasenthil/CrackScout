# CrackScout

CrackScout is a research-oriented prototype for looking at cracks in concrete images. It currently compares three classical OpenCV segmentations on the same grayscale image: inverted Otsu thresholding, adaptive thresholding, and a black-hat response.

Each saved mask is a candidate for later comparison. None of them has been evaluated against labels, so none of them is a measure of detection accuracy or a research-grade result.

## Layout

```
CrackScout/
├── README.md
├── requirements.txt
├── run_baseline.py
├── run_blackhat_sensitivity.py
├── crackscout/
│   ├── io.py            # load an image, or fail with a clear error
│   ├── preprocess.py    # grayscale and light blur
│   ├── segment.py       # registered segmentation methods
│   ├── sensitivity.py   # fixed black-hat parameter grid
│   ├── visualize.py     # side-by-side figure
│   └── pipeline.py      # wire the steps together
├── tests/
│   ├── test_baseline.py
│   └── test_blackhat_sensitivity.py
├── samples/             # put your own input image here
└── outputs/             # mask and comparison figure are written here
```

## Approach

All three methods assume a crack is darker than the surrounding concrete. That assumption is not checked against real labels. Loading, grayscale conversion, the 5x5 Gaussian blur, and saving are shared. Only the segmentation step changes.

1. Load a local image from `samples\` (or any other path you pass in).
2. Convert it to grayscale and apply a 5x5 Gaussian blur.
3. Run the method selected with `--method`. The default remains `otsu`.
4. Save a single-channel mask of 0 and 255, plus a side-by-side figure. The filename includes the method name.

| Method | What it does | Default parameters |
| --- | --- | --- |
| `otsu` | Inverted Otsu threshold, then morphological opening and closing | `open_kernel=3`, `close_kernel=5` |
| `adaptive` | Inverted adaptive threshold, so pixels darker than a local neighborhood become foreground | `block_size=35`, `C=5`, `adaptive_method="gaussian"` (`"mean"` is also available) |
| `blackhat` | Black-hat (closing minus the image) to emphasize dark features smaller than an ellipse, then a fixed cutoff | `kernel_size=15`, `threshold=25` |

`block_size` and `kernel_size` must be odd integers greater than or equal to 3. `threshold` must be an integer from 0 through 255. These defaults are explicit settings for a controlled comparison, not values selected by an accuracy study.

`segment.py` keeps the functions in a `SEGMENTERS` dictionary. The script selects one by name, and each method writes its own files:

- `outputs\<name>_otsu_mask.png` and `outputs\<name>_otsu_comparison.png`
- `outputs\<name>_adaptive_mask.png` and `outputs\<name>_adaptive_comparison.png`
- `outputs\<name>_blackhat_mask.png` and `outputs\<name>_blackhat_comparison.png`

Running `adaptive` or `blackhat` leaves an existing Otsu result in place. Running the same method again replaces only that method's two files.

Shadows, stains, joints, and rough texture can be marked as foreground. Thin or low-contrast cracks can be missed. Treat each output as an unevaluated mask until it is compared with labels.

## Setup (Windows PowerShell)

From the project directory:

```powershell
cd B:\CrackScout
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If activation is blocked by the execution policy, allow scripts for this PowerShell session only and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Python 3.10 or newer is required. Use the standard CPython `python` command above. The free-threaded build (`python3.13t`) does not currently install these binary wheels. If `py -3` points at that build, create the virtual environment with `python` instead.

The direct dependencies are NumPy, OpenCV (`opencv-python`), and Matplotlib. The commands above were run successfully with Python 3.13.7, NumPy 2.5.4, OpenCV 5.0.0.93, and Matplotlib 3.11.2.

## Add a sample image

`samples\your_crack_image.jpeg` is the input used for the current comparison. To use a different photo, copy an image you have rights to use into `samples\`:

```powershell
Copy-Item C:\path\to\your_photo.jpg .\samples\your_crack_image.jpg
```

Supported formats are `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, and `.tiff`.

## Run

Run all three methods on the same input. With the image already in `samples\`, that path is `samples\your_crack_image.jpeg`.

```powershell
python run_baseline.py --input samples\your_crack_image.jpeg --method otsu
python run_baseline.py --input samples\your_crack_image.jpeg --method adaptive
python run_baseline.py --input samples\your_crack_image.jpeg --method blackhat
```

`--method otsu` is still the default, so this writes the same Otsu filenames as the first experiment:

- `outputs\your_crack_image_otsu_mask.png`
- `outputs\your_crack_image_otsu_comparison.png`

The other two commands add:

- `outputs\your_crack_image_adaptive_mask.png`
- `outputs\your_crack_image_adaptive_comparison.png`
- `outputs\your_crack_image_blackhat_mask.png`
- `outputs\your_crack_image_blackhat_comparison.png`

Without activating the virtual environment:

```powershell
.\.venv\Scripts\python.exe run_baseline.py --input samples\your_crack_image.jpeg --method adaptive
.\.venv\Scripts\python.exe run_baseline.py --input samples\your_crack_image.jpeg --method blackhat
```

A missing path or a file OpenCV cannot decode stops the script and prints an `Error:` message.

## Black-hat parameter sensitivity

`run_blackhat_sensitivity.py` reruns the existing black-hat method on one image at five fixed settings. The current defaults stay `kernel_size=15` and `threshold=25`. The grid changes one of those at a time:

| kernel_size | threshold |
| --- | --- |
| 15 | 25 |
| 9 | 25 |
| 21 | 25 |
| 15 | 15 |
| 15 | 35 |

Preprocessing is the same grayscale conversion and `denoise()` step used by `run_baseline.py`. Each mask is still a single-channel image of 0 and 255.

The script writes a new pair for every setting, plus one text report. For `samples\your_crack_image.jpeg` the new files are:

- `outputs\your_crack_image_blackhat_k15_t25_mask.png` and `..._comparison.png`
- `outputs\your_crack_image_blackhat_k9_t25_mask.png` and `..._comparison.png`
- `outputs\your_crack_image_blackhat_k21_t25_mask.png` and `..._comparison.png`
- `outputs\your_crack_image_blackhat_k15_t15_mask.png` and `..._comparison.png`
- `outputs\your_crack_image_blackhat_k15_t35_mask.png` and `..._comparison.png`
- `outputs\your_crack_image_blackhat_sensitivity.txt`

These names are separate from `your_crack_image_blackhat_mask.png` and the Otsu and adaptive outputs. If any planned file already exists, the script stops and leaves every existing file unchanged. A later run can use another directory:

```powershell
python run_blackhat_sensitivity.py --input samples\your_crack_image.jpeg --output-dir outputs\blackhat_sensitivity_run2
```

The report lists, for each setting, the percentage of pixels equal to 255 and the number of 8-connected foreground components. White pixels can be cracks, shadows, stains, or joints. Component count changes when regions touch or break, and diagonal neighbors count as connected. The report does not measure detection accuracy, does not show whether a shadow was removed, and does not choose a setting.

```powershell
python run_blackhat_sensitivity.py --input samples\your_crack_image.jpeg
```

## Tests

The tests build temporary synthetic images. They do not need a file in `samples\`.

```powershell
python -m unittest discover -s tests -v
```

## Adding another segmentation method

In `crackscout\segment.py`, add a function and register it:

```python
def segment_example(gray: np.ndarray) -> np.ndarray:
    ...

SEGMENTERS["example"] = segment_example
```

Then run `python run_baseline.py --input samples\your_crack_image.jpg --method example`.
