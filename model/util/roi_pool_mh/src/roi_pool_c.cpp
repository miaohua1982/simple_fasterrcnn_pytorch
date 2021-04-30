#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include<cmath>

namespace py = pybind11;

int add(int i, int j) {
    return i + j;
}

inline int min(int a, int b){
    return a<b? a:b;
}

inline int max(int a, int b){
    return a<b? b:a;
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
    *buf = s;
    ++buf;
    for(int i = 1; i < n; ++i, ++buf)
        *buf = *(buf-1)+interval;
    *buf = e;
}

void roi_pooling_forward(const py::array_t<float>& feat_x, const py::array_t<float>& rois, py::array_t<float>& output_feat, py::array_t<int>& feat_pos, float spatial_scale, int roi_size) 
{
    py::buffer_info feat_x_buf = feat_x.request();
    py::buffer_info rois_buf = rois.request();
    py::buffer_info output_feat_buf = output_feat.request();
    py::buffer_info feat_pos_buf = feat_pos.request();

    if (feat_x_buf.shape[0] != 1)  // input feat's shape is assumed to be 1*ch*h*w, a typical value is 1*512*37*50
    {
        throw std::runtime_error("we only support batch==1 right now!");
    }
    if (rois_buf.shape[1] != 4)   // rois are always to be R*4, a typical value is 128*4
    {
        throw std::runtime_error("rois array shape must be match [R,4]!");
    }

    //access numpy.ndarray
    auto feats = feat_x.unchecked<4>();
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
    size_t output_feat_interval = c*roi_feat_num;
    size_t feat_pos_interval = c*roi_feat_num*2;
    for(int i = 0; i < num_rois; ++i, rois_ptr+=4) {
        Roi scale_roi(rois_ptr[0], rois_ptr[1], rois_ptr[2], rois_ptr[3]);
        scale_roi = scale_roi*spatial_scale;
        scale_roi.round();
        // generate grid axis
        gen_steps(x, scale_roi[0], scale_roi[2]+1, roi_size);
        gen_steps(y, scale_roi[1], scale_roi[3]+1, roi_size);
        // the base for buffer offset 
        int output_feat_base = i*output_feat_interval;
        int feat_pos_base = i*feat_pos_interval;
        for(int ch = 0; ch < c; ++ch) {
            int sub_feat_pos_base = ch*roi_feat_num*2;
            int sub_output_feat_base = ch*roi_feat_num;
            for(int j = 0; j < roi_size; ++j) {
                float lt_y = y[j];
                float rd_y = y[j+1];
                for(int k = 0; k < roi_size; ++k) {
                    float lt_x = x[k];
                    float rd_x = x[k+1];

                    int lt_y_pos = max(min(int(std::floor(lt_y)), h), 0);
                    int lt_x_pos = max(min(int(std::floor(lt_x)), w), 0);
                    int rd_y_pos = max(min(int(std::ceil(rd_y)), h), 0);
                    int rd_x_pos = max(min(int(std::ceil(rd_x)), w), 0);
                
                    int output_suffix_pos = roi_size*j+k;
                
                    float max_val = -1000000.0f;
                    int max_pos_y = -1;
                    int max_pos_x = -1;
                    // get the max value in [lt_x_pos:rd_x_pos) : [lt_y_pos:rd_y_pos)
                    for(int s = lt_y_pos; s < rd_y_pos; ++s)
                        for(int e = lt_x_pos; e < rd_x_pos; ++e) {
                            float tmp = feats(0, ch, s, e);
                            if(tmp>max_val) {
                                max_val = tmp;
                                max_pos_y = s;
                                max_pos_x = e;
                            }
                        }
                    // max value
                    output_feat_ptr[output_feat_base+sub_output_feat_base+output_suffix_pos] = max_val;
                    // max value' postion
                    feat_pos_ptr[feat_pos_base+sub_feat_pos_base+output_suffix_pos*2+0] = max_pos_y;
                    feat_pos_ptr[feat_pos_base+sub_feat_pos_base+output_suffix_pos*2+1] = max_pos_x;
                }
            }
        }
    }

    delete [] x;
    delete [] y;
}


void roi_pooling_backward(const py::array_t<float>& grad_output, const py::array_t<int>& feat_pos, py::array_t<float> & grad_input, int roi_size)
{
    // the buffer info for array
    py::buffer_info grad_output_buf = grad_output.request();
    py::buffer_info grad_input_buf = grad_input.request();
    // the channel & height & width for grad input buf, the same shape with feat_x in forward function
    if (grad_input_buf.shape[0] != 1)  // input grad's shape is assumed to be 1*c*h*w, a typical value is 1*512*37*50
    {
        throw std::runtime_error("we only support batch==1 right now!");
    }

    auto grad_output_cache = grad_output.unchecked<4>();
    auto feat_pos_cache = feat_pos.unchecked<4>();
    auto grad_input_cache = grad_input.mutable_unchecked<4>();

    // the batch size of grad output from upstream
    size_t bs_grad_output = grad_output_buf.shape[0];
    // the channel number of grad input
    size_t c = grad_input_buf.shape[1];

    // the base offset for buffers
    int num_feat_map = roi_size*roi_size;
    // iterator the grad one by one 
    for(int i = 0; i < bs_grad_output; ++i)
    {
        for(int ch = 0; ch < c; ++ch) {
            for(int idx = 0; idx < num_feat_map; ++idx) {
                int y = idx/roi_size;
                int x = idx%roi_size;
            
                // the feat pos has recorded the position where the max value from during forwarding procedure
                int max_feat_pos_y = feat_pos_cache(i, ch, idx, 0);
                int max_feat_pos_x = feat_pos_cache(i, ch, idx, 1);
                // add grad from upstream
                grad_input_cache(0, ch, max_feat_pos_y, max_feat_pos_x) += \
                grad_output_cache(i, ch, y, x);
            }
        }
    }
}

PYBIND11_MODULE(roi_pool_mh, m) {
    m.doc() = "roi pooling forward & backward with pybind11";

	m.attr("__version__") = "0.0.1";
    m.def("add", &add, "for test");
    m.def("roi_pooling_forward", &roi_pooling_forward, "A function do roi pooling forward operation according to input features & rois");
    m.def("roi_pooling_backward", &roi_pooling_backward, "A function do roi pooling backward operation according to grad output");
}