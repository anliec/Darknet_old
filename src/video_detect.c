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
#include <libgen.h>

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

const static int video_detect_frame = 3;
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
        axpy_cpu(video_detect_total, 1.f/video_detect_frame, detection_predictions[j], 1, avg_array, 1);
    }
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
            memcpy(l.output, avg_array + count, sizeof(float) * l.outputs);
            count += l.outputs;
        }
    }
    detection *dets = get_network_boxes(net, (int)video_width, (int)video_height, video_detect_thresh, video_detect_hier, 0, 1, nboxes);
    return dets;
}

void *detect_frame_in_thread(void *ptr)
{
    float nms = .4;

    layer l = video_detect_net->layers[video_detect_net->n-1];
    float *X = video_detect_buff_letter[(video_detect_buff_index+2)%3].data;
    show_image(video_detect_buff_letter[(video_detect_buff_index+2)%3], "image feed to yolo", 500);
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
    new_detection->next = NULL;
    new_detection->dets = dets;
    new_detection->nboxes = nboxes;
    detection_list_head->next = new_detection;
    detection_list_head = new_detection;

    video_detect_index = (video_detect_index + 1) % video_detect_frame;
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


void detections_to_rois(detection * dets, int det_count, char * rois, char * signs)
{
    int i,j;
    char is_first_sign = 1;

    for(i = 0; i < det_count; ++i){
        int class = -1;
        for(j = 0; j < video_detect_classes; ++j){
            if (dets[i].prob[j] > video_detect_thresh){
                class = j;
                break;
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

            if(is_first_sign == 1){
                is_first_sign = 0;
            }
            else{
                strcat(signs, ",");
            }
            sprintf(signs,"%s\n"
                          "                    {\"coordinates\": [%d,%d,%d,%d],\n"
                          "                     \"detection_confidence\": %f,\n"
                          "                     \"class\": \"%s\"\n"
                          "                    }",
                          signs, left, top, width, height, dets[i].prob[j], video_detect_names[class]);

            sprintf(rois, "%s%s,%d,%d,%d,%d;", rois, video_detect_names[class], left, top, width, height);
        }
    }
}

struct write_in_thread_args{
    struct detection_list_element * list_first_element;
    char * output_json_file;
    void * cap;
    char * weightsPath;
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
    time_t now;
    time (&now);
    struct tm * timeinfo;
    timeinfo = localtime (&now);
    char timeText[128];
    strftime(timeText, 128, "%A %d %B %Y, %H:%M", timeinfo);
    fprintf(json, "{\n"
                  "    \"output\": {\n"
                  "        \"video_cfg\": {\n"
                  "            \"datetime\": \"\",\n"
                  "            \"route\": \"\",\n"
                  "            \"com_pos\": \"\",\n"
                  "            \"fps\": \"%f\",\n"
                  "            \"resolution\": \"%dx%d\"\n"
                  "        },\n"
                  "        \"framework\": {\n"
                  "            \"name\": \"darknet\",\n"
                  "            \"version\": \"%s\",\n"
                  "            \"test_date\": \"%s\",\n"
                  "            \"weights\": \"%s\"\n"
                  "        },\n"
                  "        \"frames\": [\n",
            get_cap_property(args->cap, CV_CAP_PROP_FPS), (int)video_width, (int)video_height, __DATE__, timeText,
            basename(args->weightsPath));

    int frame_number = 0;

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
            char signs[1024] = "";
            detections_to_rois(cur_element->dets, cur_element->nboxes, rois, signs);

            if(frame_number != 0){
                fprintf(json, ",\n");
            }
            fprintf(json, "            {\n"
                          "                \"frame_number\": \"%07d.jpg\",\n"
                          "                \"RoIs\": \"%s\",\n"
                          "                \"signs\": [%s]\n"
                          "            }", frame_number, rois, signs);

            frame_number++;
        }
    }
    fprintf(json, "\n        ]\n"
                  "    }\n"
                  "}");

    fclose(json);
    return 0;
}

int ms_time()
{
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (int)time.tv_sec * 1000000 + (int)time.tv_usec;
}

void detect_in_video(char *cfgfile, char *weightfile, float thresh, const char *video_filename,
        char *classes_names_file, int classes_count, float hier, char *json_output_file, int decrypt_weights)
{
    video_detect_names = get_labels(classes_names_file);
    video_detect_classes = classes_count;
    video_detect_thresh = thresh;
    video_detect_hier = hier;
    printf("Video Detector\n");
    video_detect_net = load_network(cfgfile, NULL, 0);
    load_weights_encrypt(video_detect_net, weightfile, decrypt_weights);
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

    if(!cap) error("Couldn't read video file.\n");
    video_height = (float)get_cap_property(cap, CV_CAP_PROP_FRAME_HEIGHT);
    video_width = (float)get_cap_property(cap, CV_CAP_PROP_FRAME_WIDTH);

    struct write_in_thread_args writer_args;
    writer_args.list_first_element = detection_list_head;
    writer_args.output_json_file = json_output_file;
    writer_args.cap = cap;
    writer_args.weightsPath = weightfile;
    if(pthread_create(&write_thread, 0, write_in_thread, &writer_args)) error("Thread creation failed");

    video_detect_buff[0] = get_image_from_stream(cap);
    video_detect_buff[1] = copy_image(video_detect_buff[0]);
    video_detect_buff[2] = copy_image(video_detect_buff[0]);
    video_detect_buff_letter[0] = letterbox_image(video_detect_buff[0], video_detect_net->w, video_detect_net->h);
    video_detect_buff_letter[1] = letterbox_image(video_detect_buff[0], video_detect_net->w, video_detect_net->h);
    video_detect_buff_letter[2] = letterbox_image(video_detect_buff[0], video_detect_net->w, video_detect_net->h);

    int count = 0;
    int detection_time = ms_time();

    while(!video_detect_done){
        video_detect_buff_index = (video_detect_buff_index + 1) %3;
        if(pthread_create(&fetch_thread, 0, fetch_video_frame_in_thread, cap)) error("Thread creation failed");
        if(pthread_create(&detect_thread, 0, detect_frame_in_thread, 0)) error("Thread creation failed");

        int cur_time = ms_time();
        printf("                  \rFPS:%.2f",1e6/(double)(cur_time - detection_time + 1)); // prevent 0 div error
        detection_time = cur_time;

        pthread_join(fetch_thread, 0);
        pthread_join(detect_thread, 0);
        ++count;
    }

    printf("\nFinishing writing json file\n");
    pthread_join(write_thread, 0);

    // clean memory
    free_detections(detection_list_head->dets, detection_list_head->nboxes);
    close_video_stream(cap);
    free(avg_array);
    for (i = 0; i < video_detect_frame; ++i){
        free(detection_predictions[i]);
    }
    free(detection_predictions);
    free(detection_list_head);
    free_network(video_detect_net);
}

#else
void detect_in_video(char *cfgfile, char *weightfile, float thresh, const char *video_filename,
        char *classes_names_file, int classes_count, float hier, char *json_output_file)
{
    fprintf(stderr, "Demo needs OpenCV for webcam images.\n");
}
#endif

