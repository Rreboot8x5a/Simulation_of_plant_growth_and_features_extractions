
# Hi, I am a repository based on CPlantBox

## My purpose :
1. Simulate different plants through time (the Arabidopsis thaliana and the basil)
2. Simulate the view of a nadir RGB-d camera (like Kinect or Intel Realsense)
3. Extract the features of the plant through time with the camera data (actually the method only work with the basil)

## Installation guide

You just have to follow the guide of [CPlantBox](https://github.com/Plant-Root-Soil-Interactions-Modelling/CPlantBox) installation.
When done, add all the files of this repo to the root folder of your CPlantBox project.
Finally, enjoy the code (if you understand it btw)

## Important notes

- No documentation of the code, but the code is self explaining by its comments.

- As this project was done in a short period of time, the code is not ultra optimized or made cleverly. 
There is probably errors here and there that I do not even know.

- The most important thing to know before launching the code : for the time evolution script, I did not have the time to put a condition to verify if a leaf is totally hidden (it is probably easy to implement with all the things already done, but the lack of time prevent me to do it).
For BasilData and BasilData2 data, it creates fatal errors after 40 days of simulation because a leaf is totally hidden.
If you still want to see the result of this part, you just have to stop the big loop before the fatal time simulation.

- A lot of parts of this code is very tricky to explain only by words in the comments, so it will be more understandable if you take a look at the reports shared by email.

- The data used for the tests are in several folders (ArabidopsisData, BasilData, basiltest, Height, ...).

- Finally, the code in TimeEvolution works because the stem of the basil plant is always perfectly straight, but the code is easily improvable to get rid of this simplification

May the force be with you ! ;) 
