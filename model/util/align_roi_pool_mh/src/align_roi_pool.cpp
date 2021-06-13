
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include<cmath>

namespace py = pybind11;

int add(int i, int j) {
	return i + j;
}

template<class T>
inline T max(const T & a, const T & b){
    return a<b? b:a;
}

template<class T>
inline T min(const T & a, const T & b){
    return a<b? a:b;
}

void CropAndResizePerBox(
    const float * image_data, 
    const int batch_size,
    const int depth,
    const int image_height,
    const int image_width,
    const int sampling_ratio,
    const float spatial_scale,

    const float * boxes_data, 
    const int * box_index_data,
    const int limit_box,

    float * corps_data,
    const int crop_height,
    const int crop_width,
    const float extrapolation_value
) {
    const int image_channel_elements = image_height * image_width;
    const int image_elements = depth * image_channel_elements;

    const int channel_elements = crop_height * crop_width;
    const int crop_elements = depth * channel_elements;

    for (int b = 0; b < limit_box; ++b) {
        const float * box = boxes_data + b * 4;
        const float x1 = box[0]*spatial_scale;
        const float y1 = box[1]*spatial_scale;
        const float x2 = box[2]*spatial_scale;
        const float y2 = box[3]*spatial_scale;

        const int batch_ind = box_index_data[b];
        if (batch_ind < 0 || batch_ind >= batch_size) {
            char msg[100];
            sprintf(msg,"Error: batch_index %d out of range [0, %d)\n", batch_ind, batch_size);
            throw std::runtime_error(msg);
        }

        float height_scale = (crop_height > 1) ? max(y2 - y1, 1.0f) / crop_height : 0;
        float width_scale = (crop_width > 1) ? max(x2 - x1, 1.0f)  / crop_width : 0;

        int sample_ratio_height = (sampling_ratio == -1) ? std::ceilf(height_scale) : sampling_ratio;
        int sample_ratio_width = (sampling_ratio == -1) ? std::ceilf(width_scale) : sampling_ratio;

        for(int d = 0; d < depth; ++d) {
            const float *pimage = image_data + batch_ind * image_elements + d * image_channel_elements;

            for(int y = 0; y < crop_height; ++y)
            {
                const float in_y = (crop_height > 1) ? y1 + y * height_scale : 0.5f * (y1 + y2);

                if (in_y < 0 || in_y > image_height - 1)
                {
                    for (int x = 0; x < crop_width; ++x)
                    {
                        corps_data[crop_elements * b + channel_elements * d + y * crop_width + x] = extrapolation_value;
                    }
                    continue;
                }
                
                for (int x = 0; x < crop_width; ++x)
                {
                    const float in_x = (crop_width > 1)? x1 + x * width_scale : 0.5f * (x1 + x2);
                    if (in_x < 0 || in_x > image_width - 1)
                    {
                        corps_data[crop_elements * b + channel_elements * d + y * crop_width + x] = extrapolation_value;
                        continue;
                    }
                
                    float p_val = 0.0f;
                    for(int i = 0; i < sample_ratio_height; ++i) {
                        float py = in_y + height_scale*(0.5f+i)/sample_ratio_height;
                        int top_y = max(int(std::floorf(py)), 0);
                        int down_y = min(int(std::ceilf(py)), image_height-1);
                        float y_lerp = py-top_y;

                        for(int j = 0; j < sample_ratio_width; ++j) {
                            float px = in_x + width_scale*(0.5f+j)/sample_ratio_width;

                            int left_x = max(int(std::floorf(px)), 0);
                            int right_x = min(int(std::ceilf(px)), image_width-1);
                            float x_lerp = px-left_x;

                            const float top_left = pimage[top_y * image_width + left_x];
                            const float top_right = pimage[top_y * image_width + right_x];
                            const float bottom_left = pimage[down_y * image_width + left_x];
                            const float bottom_right = pimage[down_y * image_width + right_x];
                        
                            const float top = top_left + (top_right - top_left) * x_lerp;
                            const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
                            
                            p_val  += top + (bottom - top) * y_lerp;                        
                        }
                    }
                    
                    corps_data[crop_elements * b + channel_elements * d + y * crop_width + x] = p_val/sample_ratio_height/sample_ratio_width;

                }   // end for x
            }   // end for y
        } // end for depth
    }   // end for batch

}


void align_roi_pool_forward(
    const py::array_t<float>& image,  //THFloatTensor * image,
    const py::array_t<float>& boxes,  //THFloatTensor * boxes,            // [y1, x1, y2, x2]
    const py::array_t<int>& box_inds, //THIntTensor * box_index,          // range in [0, batch_size)
    const float extrapolation_value,
    const int crop_height,
    const int crop_width,
    const float spatial_scale,
    const int sampling_ratio,
    py::array_t<float> & crops        //THFloatTensor * crops
) {
  	py::buffer_info image_buf = image.request();
	py::buffer_info boxes_buf = boxes.request();
    py::buffer_info box_inds_buf = box_inds.request();
    py::buffer_info crops_buf = crops.request();

	const int batch_size = image_buf.shape[0];
    const int depth = image_buf.shape[1];
    const int image_height = image_buf.shape[2];
    const int image_width = image_buf.shape[3];

	const int num_boxes = boxes_buf.shape[0];

    // get memory access
    float* image_data = (float*)image_buf.ptr;
    float* boxes_data = (float*)boxes_buf.ptr;
    int* box_inds_data = (int*)box_inds_buf.ptr;
    float* crops_data = (float*)crops_buf.ptr;
    // crop_and_resize for each box
    CropAndResizePerBox(
        image_data,    //THFloatTensor_data(image),
        batch_size,
        depth,
        image_height,
        image_width,
        sampling_ratio,
        spatial_scale,

        boxes_data,     // THFloatTensor_data(boxes),
        box_inds_data,  // THIntTensor_data(box_index),
        num_boxes,      // the number of boxes

        crops_data,     // THFloatTensor_data(crops),
        crop_height,
        crop_width,
        extrapolation_value
    );

}

void align_roi_pool_backward(
    const py::array_t<float> & grads,   //THFloatTensor * grads,
    const py::array_t<float> & boxes,   //THFloatTensor * boxes,      // [y1, x1, y2, x2]
    const py::array_t<int> & box_inds,  //THIntTensor * box_index,    // range in [0, batch_size)
    const float spatial_scale,
    const int sampling_ratio,
    py::array_t<float> & grads_image    //THFloatTensor * grads_image // resize to [bsize, c, hc, wc]
)
{   
    py::buffer_info grads_buf = grads.request();
    py::buffer_info boxes_buf = boxes.request();
    py::buffer_info box_inds_buf = box_inds.request();
  	py::buffer_info grads_image_buf = grads_image.request();

	const int batch_size = grads_image_buf.shape[0];
    const int depth = grads_image_buf.shape[1];
    const int image_height = grads_image_buf.shape[2];
    const int image_width = grads_image_buf.shape[3];

    const int num_boxes = grads_buf.shape[0];
    const int crop_height = grads_buf.shape[2];
    const int crop_width = grads_buf.shape[3];

    // n_elements
    const int image_channel_elements = image_height * image_width;
    const int image_elements = depth * image_channel_elements;

    const int channel_elements = crop_height * crop_width;
    const int crop_elements = depth * channel_elements;

    // data pointer
    const float * grads_data = (const float*)grads_buf.ptr;
    const float * boxes_data = (const float*)boxes_buf.ptr;
    const int * box_index_data = (const int*)box_inds_buf.ptr;
    float * grads_image_data = (float*)grads_image_buf.ptr;

    for (int b = 0; b < num_boxes; ++b) {
        const float * box = boxes_data + b * 4;
        const float x1 = box[0] * spatial_scale;
        const float y1 = box[1] * spatial_scale;
        const float x2 = box[2] * spatial_scale;
        const float y2 = box[3] * spatial_scale;

        const int batch_ind = box_index_data[b];
        if (batch_ind < 0 || batch_ind >= batch_size) {
            char msg[100];
            sprintf(msg,"Error: batch_index %d out of range [0, %d)\n", batch_ind, batch_size);
            throw std::runtime_error(msg);
       }

        const float height_scale = (crop_height > 1) ? max(y2 - y1, 1.0f) / crop_height : 0;
        const float width_scale = (crop_width > 1) ? max(x2 - x1, 1.0f) / crop_width : 0;

        int sample_ratio_height = (sampling_ratio == -1) ? std::ceilf(height_scale) : sampling_ratio;
        int sample_ratio_width = (sampling_ratio == -1) ? std::ceilf(width_scale) : sampling_ratio;

        for(int d = 0; d < depth; ++d) {
            float *pimage = grads_image_data + batch_ind * image_elements + d * image_channel_elements;

            for (int y = 0; y < crop_height; ++y)
            {
                const float in_y = (crop_height > 1) ? y1 + y * height_scale : 0.5f * (y1 + y2);
                if (in_y < 0 || in_y > image_height - 1)
                {
                    continue;
                }

                for (int x = 0; x < crop_width; ++x)
                {
                    const float in_x = (crop_width > 1) ? x1 + x * width_scale : 0.5f * (x1 + x2);
                    if (in_x < 0 || in_x > image_width - 1)
                    {
                        continue;
                    }

                    float grad_val = grads_data[crop_elements * b + channel_elements * d + y * crop_width + x];
                    grad_val /= (sample_ratio_height*sample_ratio_width);

                    for(int i = 0; i < sample_ratio_height; ++i) {
                        float py = in_y + height_scale*(0.5f+i)/sample_ratio_height;
                        int top_y = max(int(std::floorf(py)), 0);
                        int down_y = min(int(std::ceilf(py)), image_height-1);
                        float y_lerp = py-top_y;

                        for(int j = 0; j < sample_ratio_width; ++j) {
                            float px = in_x + width_scale*(0.5f+j)/sample_ratio_width;
                            int left_x = max(int(std::floorf(px)), 0);
                            int right_x = min(int(std::ceilf(px)), image_width-1);
                            float x_lerp = px-left_x;

                            const float top_val = (1-y_lerp)*grad_val;
                            pimage[top_y * image_width + left_x] += (1 - x_lerp) * top_val;
                            pimage[top_y * image_width + right_x] += x_lerp * top_val;

                            const float bottem_val = y_lerp * grad_val;
                            pimage[down_y * image_width + left_x] += (1 - x_lerp) * bottem_val;
                            pimage[down_y * image_width + right_x] += x_lerp * bottem_val;
                        }
                    }

                }   // end x
            }   // end y
        } // end depth
    }   // end b
}


PYBIND11_MODULE(align_roi_pool_mh, m) {
	m.doc() = "do align roi pooling forward backward with pybind11";
	m.attr("__version__") = "0.0.1";
	m.def("add", &add, "for test");
	m.def("align_roi_pool_forward", &align_roi_pool_forward, "A function does align roi forward", \
        py::arg("image"), py::arg("boxes"), py::arg("box_inds"), py::arg("extrapolation_value"), \
        py::arg("crop_height"), py::arg("crop_width"), py::arg("spatial_scale"), py::arg("sampling_ratio"), py::arg("crops"));
	m.def("align_roi_pool_backward", &align_roi_pool_backward, "A function does align roi backward", \
		py::arg("grads"), py::arg("boxes"), py::arg("box_inds"), py::arg("spatial_scale"), py::arg("sampling_ratio"), py::arg("grads_image"));
}