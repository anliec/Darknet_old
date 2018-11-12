#include <stdlib.h>
#include <stdio.h>
#include "darknet.h"

int main(int argc, char** argv){
    if(argc != 8){
        printf("wrong argument count");
        exit(1);
    }
    int classes_count = atoi(argv[5]);
    int decrypt_weights = atoi(argv[7]);
    detect_in_video(argv[1], argv[2], 0.5, argv[3], argv[4], classes_count, 0.5, argv[6], decrypt_weights);
}
