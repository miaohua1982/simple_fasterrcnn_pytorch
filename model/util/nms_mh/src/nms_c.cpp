
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include<algorithm>
#include<vector>

namespace py = pybind11;

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

void calc_iou_helper(const float * const ptr1, const float * const ptr2, size_t size1, size_t size2, float * ptr_result) {
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

           float iou = 0.0f;
           if((area1+area2-inter_area)>0.000001f) 
              iou = inter_area/(area1+area2-inter_area);
           ptr_result[i*size1 + j] = iou;
       }
    }

}

py::array_t<float> calc_iou(const py::array_t<float>& boxes1, const py::array_t<float>& boxes2) {

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
    calc_iou_helper(ptr1, ptr2, size1, size2, ptr_result);

    return result;
}


py::array_t<int> nms(const py::array_t<float> boxes, const py::array_t<float> scores, float iou_thresh)
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

        calc_iou_helper(test_boxes, test_boxes+4, 1, size-1, boxes_iou);   // the shape is 1*boxes.shape[0]-1
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
    m.def("calc_iou", &calc_iou, "A function calculate iou between two array of boxes");
    m.def("nms", &nms, "A function do non-maximum supress for one boxes array");
}