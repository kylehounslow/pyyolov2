#include <sys/time.h>

#include "option_list.h"
#include "box.h"
#include "utils.h"
#include "demo.h"
#include "parser.h"
#include "network.h"
#include "image.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"

static network _net;
typedef struct BBox {
	float x1;
	float y1;
	float x2;
	float y2;
	float confidence;
	int cls;
} _BBox;
static int _gpu_index;
void set_gpu_index(int index);
int get_gpu_index();
void load_net(char* cfg_path, char* weights_path);
void set_batch_size(int batch_size);
void detect_objs(unsigned char* img_data, int img_width, int img_height,
		int img_channel, _BBox *return_data);
int get_max_num_bboxes();
void convert_yolo_detections(float *predictions, int classes, int num,
		int square, int side, int w, int h, float thresh, float **probs,
		box *boxes, int only_objectness);


