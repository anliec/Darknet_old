#include <stdlib.h>
#include <stdio.h>
#include "darknet.h"

int main(int argc, char** argv){
    if(argc != 7){
        printf("wrong argument count");
        exit(1);
    }
    int classes_count = atoi(argv[5]);
    detect_in_video(argv[1], argv[2], argv[3], argv[4], classes_count, argv[6]);
}
