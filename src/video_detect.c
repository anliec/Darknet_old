#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "video_detect.h"
#include <sys/time.h>
#include <unistd.h>
#include <opencv2/videoio/videoio_c.h>

#define DEMO 1

#ifdef OPENCV

static char **video_detect_names;
//static image **video_detect_alphabet;
static int video_detect_classes;

static network *video_detect_net;
static image video_detect_buff [3];
static image video_detect_buff_letter[3];
static int video_detect_buff_index = 0;
static float video_detect_thresh = 0;
static float video_detect_hier = .5;

static int video_detect_frame = 3;
static int video_detect_index = 0;
static float **detection_predictions;
static int video_detect_done = 0;
static int video_detect_total = 0;
static float *avg_array;

static float video_width = 0;
static float video_height = 0;

struct detection_list_element{
    struct detection_list_element * next;
    detection * dets;
    int nboxes;
};

struct detection_list_element * detection_list_head = NULL;

detection *avg_detection_predictions(network *net, int *nboxes)
{
    int i, j;
    int count = 0;
    fill_cpu(video_detect_total, 0, avg_array, 1);
    for(j = 0; j < video_detect_frame; ++j){
        axpy_cpu(video_detect_total, 1./video_detect_frame, detection_predictions[j], 1, avg_array, 1);
    }
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
            memcpy(l.output, avg_array + count, sizeof(float) * l.outputs);
            count += l.outputs;
        }
    }
    detection *dets = get_network_boxes(net, video_detect_buff[0].w, video_detect_buff[0].h, video_detect_thresh, video_detect_hier, 0, 1, nboxes);
    return dets;
}

void *detect_frame_in_thread(void *ptr)
{
    float nms = .4;

    layer l = video_detect_net->layers[video_detect_net->n-1];
    float *X = video_detect_buff_letter[(video_detect_buff_index+2)%3].data;
    network_predict(video_detect_net, X);

//    remember_network
    int i;
    int count = 0;
    for(i = 0; i < video_detect_net->n; ++i){
        layer layer_i = video_detect_net->layers[i];
        if(layer_i.type == YOLO || layer_i.type == REGION || layer_i.type == DETECTION){
            memcpy(detection_predictions[video_detect_index] + count, video_detect_net->layers[i].output, sizeof(float) * layer_i.outputs);
            count += layer_i.outputs;
        }
    }
    detection *dets = 0;
    int nboxes = 0;
    dets = avg_detection_predictions(video_detect_net, &nboxes);

    if (nms > 0) do_nms_obj(dets, nboxes, l.classes, nms);

    struct detection_list_element * new_detection = malloc(sizeof(struct detection_list_element));
    new_detection->dets = dets;
    new_detection->nboxes = nboxes;
    detection_list_head->next = new_detection;
    detection_list_head = new_detection;

    return 0;
}

void *fetch_video_frame_in_thread(void *cap)
{
    free_image(video_detect_buff[video_detect_buff_index]);
    video_detect_buff[video_detect_buff_index] = get_image_from_stream(cap);
    if(video_detect_buff[video_detect_buff_index].data == 0) {
        video_detect_done = 1;
        return 0;
    }
    letterbox_image_into(video_detect_buff[video_detect_buff_index], video_detect_net->w, video_detect_net->h, video_detect_buff_letter[video_detect_buff_index]);
    return 0;
}

void detections_to_rois(detection * dets, int det_count, char * rois)
{
    int i,j;

    for(i = 0; i < det_count; ++i){
        int class = -1;
        for(j = 0; j < video_detect_classes; ++j){
            if (dets[i].prob[j] > video_detect_thresh){
                class = j;
                break;
//                printf("%s: %.0f%%\n", video_detect_names[j], dets[i].prob[j]*100);
            }
        }
        if(class >= 0){
            box b = dets[i].bbox;

            int left   = (int)((b.x - (b.w / 2.f)) * video_width);
            int top    = (int)((b.y - (b.h / 2.f)) * video_height);
            int width  = (int)(b.w * video_width);
            int height = (int)(b.h * video_height);

            if(left < 0) left = 0;
            if(left + width > (int)video_width - 1) width = (int)video_width - 1 - left;
            if(top < 0) top = 0;
            if(top + height > (int)video_height - 1) height = (int)video_height - 1 - top;

            sprintf(rois, "%s%s,%d,%d,%d,%d;", rois, video_detect_names[class], left, top, width, height);
        }
    }
}

struct write_in_thread_args{
    struct detection_list_element * list_first_element;
    char * output_json_file;
    void * cap;
};

void *write_in_thread(void * raw_args)
{
    struct write_in_thread_args * args = raw_args;
    struct detection_list_element * cur_element = args->list_first_element;
    FILE *json = fopen(args->output_json_file, "w");
    if(json == NULL){
        printf("Cannot open file: '%s' !\n", args->output_json_file);
        exit(1);
    }

    // write basic header:
    fprintf(json, "{\n"
                  "    \"output\": {\n"
                  "        \"video_cfg\": {\n"
                  "            \"datetime\": \"\",\n"
                  "            \"route\": \"\",\n"
                  "            \"com_pos\": \"\",\n"
                  "            \"fps\": \"%f\",\n"
                  "            \"resolution\": \"%dx%d\"\n"
                  "        },\n"
//                  "        \"framework\": {\n"
//                  "            \"name\": \"darknet\",\n"
//                  "            \"version\": \"2018mar01\",\n"
//                  "            \"test_date\": \"30.10.2018 12:19:47\"\n"
//                  "        },\n"
                  "        \"frames\": [\n",
                  cvGetCaptureProperty(args->cap, CV_CAP_PROP_FPS), (int)video_width, (int)video_height);

    int frame_number = 1;

    while(!video_detect_done){
        if(cur_element->next == NULL){
            sleep(1); // if list already empty, sleep one second
        }
        else{
            struct detection_list_element * old_element = cur_element;
            cur_element = cur_element->next;

            //clean old element:
            free_detections(old_element->dets, old_element->nboxes);
            free(old_element);

            char rois[512] = "";
            detections_to_rois(cur_element->dets, cur_element->nboxes, rois);

            if(frame_number != 1){
                fprintf(json, ",\n");
            }
            fprintf(json, "            {\n"
                          "                \"frame_number\": \"%07d.jpg\",\n"
                          "                \"RoIs\": \"%s\"\n"
                          "            }", frame_number, rois);

            frame_number++;
        }
    }
    fprintf(json, "        ]\n"
                  "    }\n"
                  "}");

    fclose(json);
}

float ms_time()
{
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (float)time.tv_sec + (float)time.tv_usec * .000001f;
}

void detect_in_video(char *cfgfile, char *weightfile, float thresh, const char *video_filename, char **names, int classes, float hier, char *json_output_file)
{
//    image **alphabet = load_alphabet();
    video_detect_names = names;
//    video_detect_alphabet = alphabet;
    video_detect_classes = classes;
    video_detect_thresh = thresh;
    video_detect_hier = hier;
    printf("Video Detector\n");
    video_detect_net = load_network(cfgfile, weightfile, 0);
    set_batch_network(video_detect_net, 1);
    pthread_t detect_thread;
    pthread_t fetch_thread;
    pthread_t write_thread;
    detection_list_head = malloc(sizeof(struct detection_list_element));
    detection_list_head->next = NULL;
    detection_list_head->dets = NULL;
    detection_list_head->nboxes = 0;

    srand(2222222);

    int i;
    for(i = 0; i < video_detect_net->n; ++i){
        layer l = video_detect_net->layers[i];
        if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
            video_detect_total += l.outputs;
        }
    }
    detection_predictions = calloc(video_detect_frame, sizeof(float*));
    for (i = 0; i < video_detect_frame; ++i){
        detection_predictions[i] = calloc(video_detect_total, sizeof(float));
    }
    avg_array = calloc(video_detect_total, sizeof(float));
    
    printf("video file: %s\n", video_filename);
    void * cap = open_video_stream(video_filename, 0, 0, 0, 0);

    if(!cap) error("Couldn't connect to webcam.\n");
    video_height = (float)cvGetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT);
    video_width = (float)cvGetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH);

    struct write_in_thread_args writer_args;
    writer_args.list_first_element = detection_list_head;
    writer_args.output_json_file = json_output_file;
    writer_args.cap = cap;
    if(pthread_create(&write_thread, 0, write_in_thread, &writer_args)) error("Thread creation failed");

    video_detect_buff[0] = get_image_from_stream(cap);
    video_detect_buff[1] = copy_image(video_detect_buff[0]);
    video_detect_buff[2] = copy_image(video_detect_buff[0]);
    video_detect_buff_letter[0] = letterbox_image(video_detect_buff[0], video_detect_net->w, video_detect_net->h);
    video_detect_buff_letter[1] = letterbox_image(video_detect_buff[0], video_detect_net->w, video_detect_net->h);
    video_detect_buff_letter[2] = letterbox_image(video_detect_buff[0], video_detect_net->w, video_detect_net->h);

    int count = 0;
    float detection_time = ms_time();

    while(!video_detect_done){
        video_detect_buff_index = (video_detect_buff_index + 1) %3;
        if(pthread_create(&fetch_thread, 0, fetch_video_frame_in_thread, cap)) error("Thread creation failed");
        if(pthread_create(&detect_thread, 0, detect_frame_in_thread, 0)) error("Thread creation failed");

        float cur_time = ms_time();
        printf("\rFPS:%.1f",1.f/(cur_time - detection_time));
        detection_time = cur_time;

        pthread_join(fetch_thread, 0);
        pthread_join(detect_thread, 0);
        ++count;
    }

    printf("\nFinishing writing json file\n");
    pthread_join(write_thread, 0);

    free_detections(detection_list_head->dets, detection_list_head->nboxes);
}

#else
void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg_array, float hier, int w, int h, int frames, int fullscreen)
{
    fprintf(stderr, "Demo needs OpenCV for webcam images.\n");
}
#endif

