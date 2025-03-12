# Advanced_Driver_Assistance_System

This is a repo for the project of the course ECEN 5763 Embedded Computer Vision at University of Colorado Boulder.

## Introduction

Significant progress has been made in computer vision, especially in autonomous driving and Advanced Driver Assistance Systems (ADAS). These innovations enhance driver comfort, vehicle safety, and pave the way for fully autonomous cars. Embedded computer vision enables real-time processing and decision-making within vehicle systems.

The main goal of this project was to create a reliable system for vehicle and lane detection. Lane detection combined the probabilistic Hough line transform with Canny edge detection, effectively determining lane boundaries.

Initially, Haar Cascade classifiers were used for vehicle detection, but accuracy was inadequate. YOLOv4 improved detection accuracy but was impractical for real-time applications due to low frame rates from high CPU computational load.

To address this, the approach shifted to YOLOv4-tiny, a scaled-down model for higher frames per second (FPS). Although an improvement, responsiveness was still lacking. CUDNN (CUDA Deep Neural Network library) was then included to further expedite processing and improve system responsiveness and frame rate.

This study also aimed to compare the performance and outcomes of these implementations with OpenCV's CUDA implementation, evaluating GPU-accelerated processing for real-time vehicle and lane recognition tasks.

## File Organization

- All the code files are in the `code` folder.
- The report is in the file `ECEN5763_Exercise_6_Mohit_Chaudhari`.
- Video outputs are in the `outputs` folder.
- Screenshots are in the `screenshots` folder.