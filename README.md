# CS170-Project-2-Feature-Selection-with-Nearest-Neighbor

Feature selection is the process of finding and choosing which features are the most relevant given specific scenarios. A feature, according to Ivan Belcic, is “an individual measurable property or characteristic of a data point” [1]. The purpose of feature selection is to choose the best features to increase accuracy by removing “noise” or irrelevant data. For this assignment, feature selection is done with the Nearest Neighbor algorithm. Using the Nearest Neighbor algorithm, features are given a classification based on the nearest neighbor, as shown in Figure 1. Since the green dot’s closest neighbor is a red dot, it is also classified as red. 
Figure 1: Example of nearest neighbor. Notice how the green dot is closest to a red dot.

This assignment is the second project in Dr. Eamonn Keogh’s Introduction to AI course at the University of California, Riverside, during the Winter quarter of 2026 [2]. The following report details my findings for the project. In this project, I implemented three algorithms: an accuracy algorithm using Nearest Neighbor and Leave-one-out-cross-validation (LOOCV), and two search algorithms (Forward Selection and Backwards Elimination).  My language of choice for the program is C++14, and the full source code for the project is included in my repository. The link to the repository can be found in the intro and once again at the end of the report. This report compares the forward selection and backwards elimination search algorithms while using accuracy based on the nearest neighbor algorithm.

To run, use Visual Studio 2019 or any other compiler.

Output for the datasets can be found in "Text Files Containing Output"

[1] Feature Selection (by Ivan Belcic, Ibm.com)
[2] CS170 Project 1 Specifications (by Dr. Eamonn Keogh, 2026)
