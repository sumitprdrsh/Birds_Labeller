# Birds Labeller
This is a simple computer vision application program that can detect and count the number of birds flying in the sky. The source image is initially thresholded and used as an input to the algorithm that uniquely labels each pixel of different birds. The algorithm returns a grayscale image where all birds are marked with a unique grayscale value. This image is finally used to count and label the birds using different colours.


## Installation
Pre-Requisite: Python 3.1 or higher
1. Clone the source code into your local directory.
2. Create a virtual environment.
3. Install required libraries (given in requirements.txt) in the virtual environment.

```python
git clone https://github.com/sumitprdrsh/Birds_Labeller.git #For cloning the source code in local directory
pip install -r requirements.txt  #Install required libraries
```


## Execution
1. Run the below command in terminal from the projects root directory (Birds_Labeller folder).

```python
python src/main.py
```


## Usage
This program can be used in a wide variety of scenarios like automatic surveillance of birds flying over birds sanctuary. The following files can be viewed to observe how the birds are labelled.

```python
> Input image: data/birds.jpg
> Labelled grayscale image: data/Birds_Gray_labelled.jpg
> Output Image: data/Birds_Colour_labelled.jpg
```

## Open Issues and Future Scope
1. The code is not refactored yet.
2. The code is not tested on other sets of similar bird images
3. Overlapping birds in the source image are counted as one.

## Contribution
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)
