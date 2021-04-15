
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include<cmath>

namespace py = pybind11;

int add(int i, int j) {
    return i + j;
}

struct Roi
{
	float points[4];

	Roi(float ltx, float lty, float rdx, float rdy) {
		points[0] = ltx;
		points[1] = lty;
		points[2] = rdx;
		points[3] = rdy;
	}

	float operator[](int idx) {
		assert(idx >= 0 && idx <= 3);
		return points[idx];
	}

	Roi & operator *(float scale) {
		for (int i = 0; i < 4; ++i)
			points[i] = points[i] * scale;
		return *this;
	}

	Roi & operator +(float offset) {
		for (int i = 0; i < 4; ++i)
			points[i] += offset;
		return *this;
	}

	void round() {
		for (int i = 0; i < 4; ++i)
			points[i] = std::round(points[i]);
	}

};

void gen_steps(float * buf, float s, float e, int n) {
    float interval = (e-s)/n;
    for(int i = 0; i < n; ++i, ++buf)
        *buf = s+interval*i;
    *buf = e;
}

void roi_pooling_forward(const py::array_t<float>& feat_x, const py::array_t<float>& rois, py::array_t<float>& output_feat, py::array_t<int>& feat_pos, float spatial_scale, int roi_size) 
{
    py::buffer_info feat_x_buf = feat_x.request();
    py::buffer_info rois_buf = rois.request();
    py::buffer_info output_feat_buf = output_feat.request();
    py::buffer_info feat_pos_buf = feat_pos.request();

    if (feat_x_buf.shape[0] != 1)
    {
        throw std::runtime_error("we only support batch==1 right now!");
    }
    if (rois_buf.shape[1] != 4))
    {
        throw std::runtime_error("rois array shape must be match [R,4]!");
    }

    //access numpy.ndarray
    float* feat_ptr = (float*)feat_x_buf.ptr;
    float* rois_ptr = (float*)rois_buf.ptr;
    float* output_feat_ptr = (float*)output_feat_buf.ptr;
    int* feat_pos_ptr = (int*)feat_pos_buf.ptr;

    size_t c = feat_x_buf.shape[1];
    size_t h = feat_x_buf.shape[2];
    size_t w = feat_x_buf.shape[3];
    size_t num_rois = rois_buf.shape[0];

    float * x = new float[roi_size+1]();
    float * y = new float[roi_size+1]();

    int roi_feat_num = roi_size*roi_size;
    int feat_interval = h*w;
    int output_feat_interval = c*roi_feat_num;
    int feat_pos_interval = roi_feat_num*c*2;
    for(int i = 0; i < num_rois; ++i) {
        Roi scale_roi(rois_ptr[0], rois_ptr[1], rois_ptr[2], rois_ptr[3]);
        scale_roi = scale_roi*spatial_scale;
        scale_roi.round();
        // generate grid axis
        gen_steps(x, scale_roi[0], scale_roi[2]+1, roi_size);
        gen_steps(y, scale_roi[1], scale_roi[3]+1, roi_size);
        
        int output_feat_base = i*output_feat_interval;
        int feat_pos_base = i*feat_pos_interval;
        for(int j = 0; j < roi_size; ++j) {
            float lt_y = y[j];
            float rd_y = y[j+1];
            for(int k = 0; k < roi_size; ++k) {
                float lt_x = x[k];
                float rd_x = x[k+1];

                lt_y_pos = std::max(std::min(int(std::floor(lt_y)), h), 0);
                lt_x_pos = std::max(std::min(int(std::floor(lt_x)), w), 0);
                rd_y_pos = std::max(std::min(int(std::ceil(rd_y)), h), 0);
                rd_x_pos = std::max(std::min(int(std::ceil(rd_x)), w), 0);

                // get the max value in [lt_x_pos:rd_x_pos) : [lt_y_pos:rd_y_pos)
                
                int output_suffix_pos = roi_size*j+k;
                int output_feat_pos_base = output_suffix_pos*c*2;
                for(int ch = 0; ch < c; ++ch) {
                    float max_val = -1000000.0f;
                    int max_pos_y = -1;
                    int max_pos_x = -1
                    for(int s = lt_y_pos; s < rd_y_pos; ++s)
                        for(int e = lt_x_pos; e < rd_x_pos; ++e)
                            if(feat_ptr[ch*feat_interval+w*s+e]>max_val) {
                                max_val = feat_ptr[ch*feat_interval+w*s+e];
                                max_pos_y = s;
                                max_pos_x = e;
                            }
                    
                    // max value
                    output_feat_ptr[output_feat_base+ch*roi_feat_num+output_suffix_pos] = max_val;
                    // max value' postion
                    feat_pos_ptr[feat_pos_base+output_feat_pos_base+ch*2+0] = max_pos_y;
                    feat_pos_ptr[feat_pos_base+output_feat_pos_base+ch*2+1] = max_pos_x;
                }
            }
        }
    }

    delete [] x;
    delete [] y;
}


void roi_pooling_backward(const py::array_t<float>& grad_output, const py::array_t<int>& feat_pos, py::array_t<float> & grad_input, int roi_size)
{
    py::buffer_info grad_output_buf = grad_output.request();
    py::buffer_info grad_input_buf = grad_input.request();
    py::buffer_info feat_pos_buf = feat_pos.request();

    //access numpy.ndarray
    float* grad_input_ptr = (float*)grad_input_buf.ptr;
    float* grad_output_ptr = (float*)grad_output_buf.ptr;
    int* feat_pos_ptr = (int*)feat_pos_buf.ptr;

    size_t c = grad_input_buf.shape[1];
    size_t h = grad_input_buf.shape[2];
    size_t w = grad_input_buf.shape[3];

    size_t bs = grad_output_buf.shape[0];
    int num_feat_map = roi_size*roi_size;
    int feat_pos_inter = roi_size*roi_size*c*2;
    int output_grad_inter = c*roi_size*roi_size;
    int one_output_grad_inter = roi_size*roi_size;
    int one_feat_pos_inter = c*2;
    int one_input_feat_inter = h*w;
    for(int i = 0; i < bs; ++i)
    {
        int feat_pos_base = i*feat_pos_inter;
        int output_grad_base = i*output_grad_inter;
        for(int idx = 0; idx < num_feat_map; ++idx) {
            int pos_y = idx/roi_size;
            int pos_x = idx%roi_size;
            
            int one_feat_pos_base = idx*one_feat_pos_inter;
            for(int ch = 0; ch < c; ++ch) {
                int max_feat_pos_y = feat_pos_ptr[feat_pos_base+one_feat_pos_base+ch*2+0];
                int max_feat_pos_x = feat_pos_ptr[feat_pos_base+one_feat_pos_base+ch*2+1];

                grad_input_ptr[ch*one_input_feat_inter+max_feat_pos_y*w+max_feat_pos_x] += \
                grad_output_ptr[output_grad_base+ch*one_output_grad_inter+idx]
            }
        }
    }
}

PYBIND11_MODULE(roi_pool_mh, m) {
    m.doc() = "roi pooling forward & backward with pybind11";

    m.def("add", &add, "for test");
    m.def("roi_pooling_forward", &roi_pooling_forward, "A function do roi pooling forward operation according to input features & rois");
    m.def("roi_pooling_backward", &roi_pooling_backward, "A function do roi pooling backward operation according to grad output");
}