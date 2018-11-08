# SSAD43
## 3D reconstruction project


## Goal
The goal of the project is to 3D reconstruct videos in real time using real time footage from different cameras

## Setup

Step 1: Clone the repo  
Step 2: Create a virtual env  
Step 3: Activate the virtual env  
Step 4: `pip3 install -r requirements.txt`  
Check whether open cv and numpy are installed

## Running the code

To run the code  
`python3 stitch.py -f (path of pic 1) -s (path of pic2)` 

## File Overviews
       .
        ├── data
        ├── Design Doc.doc
        ├── documents
        ├── meeting_minutes
        ├── our_results
        ├── README.md
        ├── requirements.txt
        ├── src
        ├── SRS.doc
        └── TestPlan.xls

### Documents
Contains the basic documentation

### Our Results
Contains the results of running algorithms on multiple images

### Src
Contains the code 

    
## Progress so far
Initially we were using homography matrix for transformation. However that approach turned out to be wrong as the transformation wasnt able to take depth into account. Now for multiple viewpoints we are using essential matrix to find out the transfromation in order to stitch them.

