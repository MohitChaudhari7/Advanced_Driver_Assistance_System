The makefile is required for all compilation.
For running any program, use cars_roadway.mp4 as input
standard execution command example: sudo ./lane_haar <path to cars_roadway.mp4> -s
Note that for execution of yolo based files, the (.weights,.cfg and coco.names) files are required from yolov4 or yolov4-tiny

ffmpeg command:
ffmpeg -framerate 25 -i frame%04d.jpg -c:v libx264 -r 25 ~/ECV/final/cudnn_yolo_lane.mp4
