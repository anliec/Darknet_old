#ifndef DEMO_H
#define DEMO_H

#include "image.h"

void detect_in_video(char *cfgfile, char *weightfile, float thresh, const char *video_filename, char **names, int classes, float hier, char *json_output_file);

#endif
