
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include<algorithm>
#include<cmath>
#include<vector>
#include<string>
#include<map>

namespace py = pybind11;

#define PI 3.14159265f

int add(int i, int j) {
    return i + j;
}

inline float min(float a, float b){
    return a<b? a:b;
}

inline float max(float a, float b){
    return a<b? b:a;
}

std::vector<int> reverse_argsort(const float * array, size_t size)
{
	std::vector<int> array_index(size, 0);
	for (int i = 0; i < size; ++i)
		array_index[i] = i;

	std::sort(array_index.begin(), array_index.end(),
		[&array](int pos1, int pos2) {return (array[pos1] > array[pos2]);});

	return array_index;
}

void calc_iou_iou(const float * const ptr1, const float * const ptr2, size_t size1, size_t size2, float * ptr_result) {
    for(int i = 0; i < size1; ++i) {
        float left_top_x1 = ptr1[i*4];
        float left_top_y1 = ptr1[i*4+1];
        float right_down_x1 = ptr1[i*4+2];
        float right_down_y1 = ptr1[i*4+3];
        float area1 = (right_down_x1-left_top_x1)*(right_down_y1-left_top_y1);

       for(int j = 0; j < size2; ++j) {
           float left_top_x2 = ptr2[j*4];
           float left_top_y2 = ptr2[j*4+1];
           float right_down_x2 = ptr2[j*4+2];
           float right_down_y2 = ptr2[j*4+3];
           float area2 = (right_down_x2-left_top_x2)*(right_down_y2-left_top_y2);

           float lt_x = max(left_top_x1, left_top_x2);
           float lt_y = max(left_top_y1, left_top_y2);
           float rd_x = min(right_down_x1, right_down_x2);
           float rd_y = min(right_down_y1, right_down_y2);

           float ws = max(0.0f, rd_x-lt_x);
           float hs = max(0.0f, rd_y-lt_y);
           float inter_area = ws*hs;

           float iou = inter_area/(area1+area2-inter_area+0.0000001f);
           ptr_result[i*size2 + j] = iou;
       }
    }
}

void calc_iou_giou(const float * const ptr1, const float * const ptr2, size_t size1, size_t size2, float * ptr_result) {
    for(int i = 0; i < size1; ++i) {
        float left_top_x1 = ptr1[i*4];
        float left_top_y1 = ptr1[i*4+1];
        float right_down_x1 = ptr1[i*4+2];
        float right_down_y1 = ptr1[i*4+3];
        float area1 = (right_down_x1-left_top_x1)*(right_down_y1-left_top_y1);

       for(int j = 0; j < size2; ++j) {
           float left_top_x2 = ptr2[j*4];
           float left_top_y2 = ptr2[j*4+1];
           float right_down_x2 = ptr2[j*4+2];
           float right_down_y2 = ptr2[j*4+3];
           float area2 = (right_down_x2-left_top_x2)*(right_down_y2-left_top_y2);

           // calc the iou & union area
           float lt_x = max(left_top_x1, left_top_x2);
           float lt_y = max(left_top_y1, left_top_y2);
           float rd_x = min(right_down_x1, right_down_x2);
           float rd_y = min(right_down_y1, right_down_y2);

           float ws = max(0.0f, rd_x-lt_x);
           float hs = max(0.0f, rd_y-lt_y);
           float inter_area = ws*hs;

           float union_area = area1+area2-inter_area+0.000001f;  // avoid to devide zero
           float iou = inter_area/union_area;
    
           // clac the minimum box area
           lt_x = min(left_top_x1, left_top_x2);
           lt_y = min(left_top_y1, left_top_y2);
           rd_x = max(right_down_x1, right_down_x2);
           rd_y = max(right_down_y1, right_down_y2);

           float con_area = (rd_x-lt_x)*(rd_y-lt_y)+0.000001f;   // avoid to devide zero
           float giou = iou-(con_area-(area1+area2-inter_area))/con_area;

           ptr_result[i*size2 + j] = giou;
       }
    }

}

void calc_iou_diou(const float * const ptr1, const float * const ptr2, size_t size1, size_t size2, float * ptr_result) {
    for(int i = 0; i < size1; ++i) {
        float left_top_x1 = ptr1[i*4];
        float left_top_y1 = ptr1[i*4+1];
        float right_down_x1 = ptr1[i*4+2];
        float right_down_y1 = ptr1[i*4+3];
        float area1 = (right_down_x1-left_top_x1)*(right_down_y1-left_top_y1);

        float center_x1 = (left_top_x1+right_down_x1)/2;
        float center_y1 = (left_top_y1+right_down_y1)/2;
        for(int j = 0; j < size2; ++j) {
           float left_top_x2 = ptr2[j*4];
           float left_top_y2 = ptr2[j*4+1];
           float right_down_x2 = ptr2[j*4+2];
           float right_down_y2 = ptr2[j*4+3];
           float area2 = (right_down_x2-left_top_x2)*(right_down_y2-left_top_y2);

           float center_x2 = (left_top_x2+right_down_x2)/2;
           float center_y2 = (left_top_y2+right_down_y2)/2;

           // calc the iou
           float lt_x = max(left_top_x1, left_top_x2);
           float lt_y = max(left_top_y1, left_top_y2);
           float rd_x = min(right_down_x1, right_down_x2);
           float rd_y = min(right_down_y1, right_down_y2);

           float ws = max(0.0f, rd_x-lt_x);
           float hs = max(0.0f, rd_y-lt_y);
           float inter_area = ws*hs;

           float union_area = area1+area2-inter_area+0.000001f;  // avoid to devide zero
           float iou = inter_area/union_area;
    
           // calc the distance between center points of two boxes
           float dis_center = (center_x2-center_x1)*(center_x2-center_x1)+ (center_y2-center_y1)*(center_y2-center_y1);
           // clac the minimum box diagonal distance
           lt_x = min(left_top_x1, left_top_x2);
           lt_y = min(left_top_y1, left_top_y2);
           rd_x = max(right_down_x1, right_down_x2);
           rd_y = max(right_down_y1, right_down_y2);

           float dis_con = (rd_x-lt_x)*(rd_x-lt_x)+(rd_y-lt_y)*(rd_y-lt_y)+0.000001f; // avoid to devide zero
           
           // the value of d-iou
           float diou = iou-dis_center/dis_con;

           ptr_result[i*size2 + j] = diou;
       }
    }
}

void calc_iou_ciou(const float * const ptr1, const float * const ptr2, size_t size1, size_t size2, float * ptr_result) {
    for(int i = 0; i < size1; ++i) {
        float left_top_x1 = ptr1[i*4];
        float left_top_y1 = ptr1[i*4+1];
        float right_down_x1 = ptr1[i*4+2];
        float right_down_y1 = ptr1[i*4+3];

        float w1 = right_down_x1-left_top_x1;
        float h1 = right_down_y1-left_top_y1;
        float area1 = w1*h1;

        float center_x1 = (left_top_x1+right_down_x1)/2;
        float center_y1 = (left_top_y1+right_down_y1)/2;
        float arctan1 = std::atan(w1/h1);
        for(int j = 0; j < size2; ++j) {
           float left_top_x2 = ptr2[j*4];
           float left_top_y2 = ptr2[j*4+1];
           float right_down_x2 = ptr2[j*4+2];
           float right_down_y2 = ptr2[j*4+3];

           float w2 = right_down_x2-left_top_x2;
           float h2 = right_down_y2-left_top_y2;
           float area2 = w2*h2;

           float center_x2 = (left_top_x2+right_down_x2)/2;
           float center_y2 = (left_top_y2+right_down_y2)/2;
           float arctan2 = std::atan(w2/h2);

           // calc the iou
           float lt_x = max(left_top_x1, left_top_x2);
           float lt_y = max(left_top_y1, left_top_y2);
           float rd_x = min(right_down_x1, right_down_x2);
           float rd_y = min(right_down_y1, right_down_y2);

           float ws = max(0.0f, rd_x-lt_x);
           float hs = max(0.0f, rd_y-lt_y);
           float inter_area = ws*hs;

           float union_area = area1+area2-inter_area+0.000001f;  // avoid to devide zero
           float iou = inter_area/union_area;
    
           // calc the distance between center points of two boxes
           float dis_center = (center_x2-center_x1)*(center_x2-center_x1)+ (center_y2-center_y1)*(center_y2-center_y1);
           // clac the minimum box diagonal distance
           lt_x = min(left_top_x1, left_top_x2);
           lt_y = min(left_top_y1, left_top_y2);
           rd_x = max(right_down_x1, right_down_x2);
           rd_y = max(right_down_y1, right_down_y2);

           float dis_con = (rd_x-lt_x)*(rd_x-lt_x)+(rd_y-lt_y)*(rd_y-lt_y)+0.000001f; // avoid to devide zero
           
           // calc the w/h scale
           float arctan = arctan2 - arctan1;  //np.arctan(w_gt / h_gt) - np.arctan(w / h)
           float v = 4.0f/PI/PI*arctan*arctan;
           float alpha = v/(1-iou+v);

           // ciou = iou - dis_boxes/dis_con-alpha*v
           // the value of c-iou
           float ciou = iou-dis_center/dis_con-alpha*v;

           ptr_result[i*size2 + j] = ciou;
       }
    }
}


typedef void(*iou_calc)(const float * const, const float * const, size_t, size_t, float *);

py::array_t<float> calc_iou(const py::array_t<float>& boxes1, const py::array_t<float>& boxes2, const std::string iou_algo="iou") {

    py::buffer_info boxes1_buf = boxes1.request();
    py::buffer_info boxes2_buf = boxes2.request();

    if ((boxes1_buf.shape[1] != 4) || (boxes2_buf.shape[1] != 4))
    {
        throw std::runtime_error("boxes array shape must be match [R,4]!");
    }

    //access numpy.ndarray
    float* ptr1 = (float*)boxes1_buf.ptr;
    float* ptr2 = (float*)boxes2_buf.ptr;

    size_t size1 = boxes1_buf.shape[0];
    size_t size2 = boxes2_buf.shape[0];

    //apply for new memory
    auto result = py::array_t<float>(size1*size2);
    //reshape to 2d array
    result.resize({size1,size2});
    py::buffer_info buf_result = result.request();
    float * ptr_result = (float*)buf_result.ptr;
    // calc iou
    // iou ~[0,1]
    // giou ~[-1,1]
    // diou ~[-1,1]
    // ciou ~[-1,1]
    std::map<std::string, iou_calc> iou_routine;
    iou_routine["iou"] = calc_iou_iou;
    iou_routine["giou"] = calc_iou_giou;
    iou_routine["ciou"] = calc_iou_ciou;
    iou_routine["diou"] = calc_iou_diou;
    iou_routine[iou_algo](ptr1, ptr2, size1, size2, ptr_result);
    // result shape is size of pb * size of gt
    return result;
}

py::array_t<int> nms(const py::array_t<float> boxes, const py::array_t<float> scores, float iou_thresh=0.5, const std::string iou_algo="iou")
{
    py::buffer_info boxes_buf = boxes.request();
    py::buffer_info scores_buf = scores.request();

    if (boxes_buf.shape[1] != 4)
    {
        throw std::runtime_error("boxes array shape must be match [R,4]!");
    }
    
    if (boxes_buf.shape[0] != scores_buf.shape[0])
    {
        throw std::runtime_error("number of boxes must be equal to number of scores!");
    }


    std::map<std::string, iou_calc> iou_routine;
    iou_routine["iou"] = calc_iou_iou;
    iou_routine["giou"] = calc_iou_giou;
    iou_routine["ciou"] = calc_iou_ciou;
    iou_routine["diou"] = calc_iou_diou;
    
    // access numpy.ndarray
    float* boxes_ptr = (float*)boxes_buf.ptr;
    float* score_ptr = (float*)scores_buf.ptr;
    size_t size = scores_buf.shape[0];
    // no box
    if(size<1)
        return py::array_t<int>();
    // reverse argsort
    std::vector<int> arg_idx = reverse_argsort(score_ptr, size);
    std::vector<int> result;
    float * test_boxes = new float[size*4]();
    float * boxes_iou = new float[size]();
    while(size>=2){
        for(int i = 0; i < size; ++i)
            memcpy(&test_boxes[i*4], &boxes_ptr[arg_idx[i]*4], 4*sizeof(float));
        // test_boxes+4: means the remaining boxes, every box has 4 points(x1,y1,x2,y2)
        iou_routine[iou_algo](test_boxes, test_boxes+4, 1, size-1, boxes_iou);   // the shape is 1*boxes.shape[0]-1
        // keep first one
        result.push_back(arg_idx[0]);
        // get remainder keep list
        size -= 1;
        int counter = 0;
        for(int i = 0; i < size; ++i) {
            if(boxes_iou[i]<=iou_thresh) {
                arg_idx[counter++] = arg_idx[i+1];  // start from 1, 0 is biggest score
            }
        }
        size = counter;
    }

    delete [] test_boxes;
    delete [] boxes_iou;
    // add last one if need
    if(size>0)
        result.push_back(arg_idx[0]);

    //apply for new memory for result
    size = result.size();
    auto keep_list = py::array_t<int>(size);
    py::buffer_info buf_result = keep_list.request();
    int * ptr_result = (int*)buf_result.ptr;
    for(int i = 0; i < size; ++i)
        ptr_result[i] = result[i];
    return keep_list;
}

PYBIND11_MODULE(nms_mh, m) {
    m.doc() = "non-maximum supress with pybind11";

    m.def("add", &add, "for test");
    m.def("calc_iou", &calc_iou, "A function calculate iou between two array of boxes", py::arg("boxes1"), py::arg("boxes2"), py::arg("iou_algo")="iou");
    m.def("nms", &nms, "A function do non-maximum supress for one boxes array", \
          py::arg("boxes"), py::arg("scores"), py::arg("iou_thresh")=0.5, py::arg("iou_algo")="iou");
}
