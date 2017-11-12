#include "pyyolov2.h"
#include <cuda.h>

void load_net(char* cfg_path, char* weights_path) {

	_net = parse_network_cfg(cfg_path);
	_net.gpu_index = _gpu_index; // set by user. default = 0
	load_weights(&_net, weights_path);
	set_batch_size(1);

}

void set_gpu_index(int index){
	_gpu_index = index; // for setting _net->gpu_index
	cudaSetDevice(index);
	struct cudaDeviceProp prop;
	cudaError_t cuda_error = cudaGetDeviceProperties(&prop, index);
	check_error(cuda_error);
	int device;
	cuda_error = cudaGetDevice(&device);
	check_error(cuda_error);
	printf("CUDA device set to %d:%s\n", device, prop.name);

}

int get_gpu_index(){
	int count;
	cudaError_t cuda_error = cudaGetDeviceCount(&count);
	check_error(cuda_error);
	return count;
}

int get_net_width() {

	return _net.w;
}

int get_net_height() {

	return _net.h;
}

void set_batch_size(int batch_size) {

	set_batch_network(&_net, batch_size);
	srand(2222222);
}

//convert numpy array passed in as unsigned char pointer to darknet's "image" datatype
image _uchar_p_to_image(unsigned char* data, int h, int w, int c) {
	image out = make_image(w, h, c);
	int count = 0;

	for (int channel = 0; channel < c; channel++) {
		for (int ind = 0; ind < h * w; ind++) {
			out.data[count++] = (float) data[ind * c + channel] / 255.;

		}
	}
	return out;
}

void test_show_image(unsigned char* img_data, int img_width, int img_height,
		int img_channel) {

	image im = _uchar_p_to_image(img_data, img_height, img_width, img_channel);
	show_image(im, "test");
	cvWaitKey(1);

}

void _convert_yolo_detections(image im, int num, box *boxes, float **probs,
		int num_classes, _BBox *return_data) {
	// convert x,y,w,h format to x1,y1,x2,y2 format
	// also include confidence and class label in return data
	for (int i = 0; i < num; ++i) {
		int class = max_index(probs[i], num_classes);
		float prob = probs[i][class];
		box b = boxes[i];
		// scale back predicted bboxes to size of input image
		float left = (b.x - b.w / 2.) * im.w;
		float right = (b.x + b.w / 2.) * im.w;
		float top = (b.y - b.h / 2.) * im.h;
		float bot = (b.y + b.h / 2.) * im.h;

		if (left < 0)
			left = 0;
		if (right > im.w - 1)
			right = im.w - 1;
		if (top < 0)
			top = 0;
		if (bot > im.h - 1)
			bot = im.h - 1;

		_BBox bb;

		bb.x1 = left;
		bb.y1 = top;
		bb.x2 = right;
		bb.y2 = bot;
		bb.confidence = prob;
		bb.cls = class;
		return_data[i] = bb;

	}

}

int get_max_num_bboxes() {

	layer l = _net.layers[_net.n - 1];
	return l.w * l.h * l.c;
}

//TODO: learn to pass in numpy array
void detect_objs(unsigned char* img_data, int img_width, int img_height,
		int img_channel, _BBox *return_data) {

	if (img_width != _net.w || img_height != _net.h) {
		printf("Input image dimensions do not match network.\n");
		printf("%d,%d (img) != %d,%d (net)\n", img_width, img_height, _net.w,
				_net.h);
	}
	image im = _uchar_p_to_image(img_data, img_height, img_width, img_channel);
	float *X = im.data;
	network_predict(_net, X);
	layer det_layer = _net.layers[_net.n - 1];
	static box *boxes;
	static float **probs;
	boxes = (box *) calloc(det_layer.w * det_layer.h * det_layer.n,
			sizeof(box));
//	printf("allocated %d bytes for boxes\n",det_layer.w * det_layer.h * det_layer.n,
//			sizeof(box));
	probs = (float **) calloc(det_layer.w * det_layer.h * det_layer.n,
			sizeof(float *));
	for (int j = 0; j < det_layer.w * det_layer.h * det_layer.n; ++j)
		probs[j] = calloc(det_layer.classes, sizeof(float *));

	float nms = .4f;
	float demo_hier = .5f;
	get_region_boxes(det_layer, im.w, im.h, _net.w, _net.h, 0, probs, boxes, 0,
			0, demo_hier, 1);
//	get_detection_boxes(det_layer, 1, 1, detection_threshold, probs, boxes, 0);
	if (nms > 0)
		do_nms_obj(boxes, probs, det_layer.w * det_layer.h * det_layer.n,
				det_layer.classes, nms);

	_convert_yolo_detections(im, det_layer.w * det_layer.h * det_layer.n, boxes,
			probs, det_layer.classes, return_data);

	free_image(im);
	free(boxes);
	for (int j = 0; j < det_layer.w * det_layer.h * det_layer.n; j++) {
		free(probs[j]);
	}
	free(probs);
}

void run_yolo_demo(char *cfg, char *datacfg, char *weights, float thresh) {

	char *prefix = 0;
//    float thresh = find_float_arg(argc, argv, "-thresh", .2);
	int cam_index = 0;
	int frame_skip = 0;
	char *filename = 0;
	float hier_thresh = 0.5;
	int width = 0, height = 0, fps = 0;
	int fullscreen = 0;
	list *options = read_data_cfg(datacfg);
	int classes = 80;
	char *name_list = option_find_str(options, "names", "data/names.list");
	char **names = get_labels(name_list);
	demo(cfg, weights, thresh, cam_index, filename, names, classes, frame_skip,
			prefix, hier_thresh, width, height, fps, fullscreen);

}

//float THRESHOLD = 0.4;

CvTrackbarCallback trackbar_callback(int pos) {

//	THRESHOLD = pos / 100.f;
	return 0;
}

void rtsp_test(char* rtsp_addr) {
	const char* window_name = "RTSP Test";
	cvNamedWindow(window_name, CV_WINDOW_AUTOSIZE);
	const char* trackbar_name = "threshold";

	int thresh_int = 40;
	int count = 100;
	cvCreateTrackbar(trackbar_name, window_name, &thresh_int, count,
			trackbar_callback);

	CvCapture *_cap = cvCaptureFromFile(rtsp_addr);
	char* cfg_path = "../cfg/yolo.cfg";
	char* weights_path = "../yolo.weights";
	load_net(cfg_path, weights_path);
	layer l = _net.layers[_net.n - 1];
	_BBox *bboxes = (_BBox *) calloc(l.w * l.h * l.n, sizeof(_BBox));

	while (1) {
		IplImage* img = cvQueryFrame(_cap);
		IplImage *img_resize = cvCreateImage(cvSize(_net.w, _net.h), img->depth,
				img->nChannels);
		cvResize(img, img_resize, CV_INTER_LINEAR);
		float scale_factor_x = (float) _net.w / img->width;
		float scale_factor_y = (float) _net.h / img->height;

		detect_objs(img_resize->imageData, img_resize->width,
				img_resize->height, img_resize->nChannels, bboxes);

		for (int i = 0; i < l.w * l.h * l.n; i++) {

			_BBox bb = bboxes[i];
			if (bb.confidence > (float) thresh_int / 100.f) {
				if (bb.cls == 0) {
//				printf("bb=%f,%f,%f,%f, cfd=%f, cls=%d(PERSON)\n", bb.x1, bb.y1,
//						bb.x2, bb.y2, bb.confidence, bb.cls);
					cvRectangle(img,
							cvPoint(bb.x1 / scale_factor_x,
									bb.y1 / scale_factor_y),
							cvPoint(bb.x2 / scale_factor_x,
									bb.y2 / scale_factor_y),
							cvScalar(0, 255, 0, 0), 2, 1, 0);
					CvFont font = cvFont(1.f, 2);
					cvPutText(img, "PERSON",
							cvPoint(bb.x1 / scale_factor_x,
									bb.y1 / scale_factor_y), &font,
							cvScalar(0, 255, 0, 0));

				} else if (bb.cls == 2 || bb.cls == 3 || bb.cls == 4
						|| bb.cls == 6 || bb.cls == 8) {
					//				printf("bb=%f,%f,%f,%f, cfd=%f, cls=%d(PERSON)\n", bb.x1, bb.y1,
					//						bb.x2, bb.y2, bb.confidence, bb.cls);
					cvRectangle(img,
							cvPoint(bb.x1 / scale_factor_x,
									bb.y1 / scale_factor_y),
							cvPoint(bb.x2 / scale_factor_x,
									bb.y2 / scale_factor_y),
							cvScalar(0, 0, 255, 0), 2, 1, 0);
					CvFont font = cvFont(1.f, 2);
					cvPutText(img, "VEHICLE",
							cvPoint(bb.x1 / scale_factor_x,
									bb.y1 / scale_factor_y), &font,
							cvScalar(0, 0, 255, 0));

				}
//			printf("bb=%f,%f,%f,%f, cfd=%f, cls=%d\n", bb.x1, bb.y1, bb.x2,
//					bb.y2, bb.confidence, bb.cls);
			}
		}

		cvShowImage(window_name, img);
		int key = cvWaitKey(1) & 0xFF;
		if (key == 27) {
			return;
		}
//	cvReleaseImage(&img);
		cvReleaseImage(&img_resize);
	}
	cvReleaseCapture(&_cap);

}

