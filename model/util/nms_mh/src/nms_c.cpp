
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>

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

py::array_t<float> calc_iou(py::array_t<float>& boxes1, py::array_t<float>& boxes2) {

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

    return result;

}

PYBIND11_MODULE(nms, m) {
    m.doc() = "non-maximum supress with pybind11";

    m.def("add", &add, "for test");
    m.def("calc_iou", &calc_iou, "A function calculate iou between two array of boxes");
}