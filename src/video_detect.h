#ifndef DEMO_H
#define DEMO_H

#include "image.h"

void detect_in_video(char *cfgfile, char *weightfile, float thresh, const char *video_filename, char *classes_names_file,
        int classes_count, float hier, char *json_output_file, int decrypt_weights);

#endif
