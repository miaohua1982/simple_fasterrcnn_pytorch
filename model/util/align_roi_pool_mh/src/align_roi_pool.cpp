
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include<cmath>

namespace py = pybind11;

int add(int i, int j) {
	return i + j;
}

void CropAndResizePerBox(
    const float * image_data, 
    const int batch_size,
    const int depth,
    const int image_height,
    const int image_width,
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
        const float x1 = (box[0]-0.5)*spatial_scale;
        const float y1 = (box[1]-0.5)*spatial_scale;
        const float x2 = (box[2]-0.5)*spatial_scale;
        const float y2 = (box[3]-0.5)*spatial_scale;

        const int b_in = box_index_data[b];
        if (b_in < 0 || b_in >= batch_size) {
            char msg[100];
            sprintf(msg,"Error: batch_index %d out of range [0, %d)\n", b_in, batch_size);
            throw std::runtime_error(msg);
        }

        const float height_scale =
            (crop_height > 1) ? (y2 - y1) / (crop_height - 1) : 0;
        const float width_scale =
            (crop_width > 1) ? (x2 - x1)  / (crop_width - 1) : 0;

        for (int y = 0; y < crop_height; ++y)
        {
            const float in_y = (crop_height > 1)
                                   ? y1 + y * height_scale
                                   : 0.5f * (y1 + y2);

            if (in_y < 0 || in_y > image_height - 1)
            {
                for (int x = 0; x < crop_width; ++x)
                {
                    for (int d = 0; d < depth; ++d)
                    {
                        // crops(b, y, x, d) = extrapolation_value;
                        corps_data[crop_elements * b + channel_elements * d + y * crop_width + x] = extrapolation_value;
                    }
                }
                continue;
            }
            
            const int top_y_index = std::floor(in_y);
            const int bottom_y_index = std::ceil(in_y);
            const float y_lerp = in_y - top_y_index;

            for (int x = 0; x < crop_width; ++x)
            {
                const float in_x = (crop_width > 1)
                                       ? x1 + x * width_scale
                                       : 0.5f * (x1 + x2);
                if (in_x < 0 || in_x > image_width - 1)
                {
                    for (int d = 0; d < depth; ++d)
                    {
                        corps_data[crop_elements * b + channel_elements * d + y * crop_width + x] = extrapolation_value;
                    }
                    continue;
                }
            
                const int left_x_index = std::floor(in_x);
                const int right_x_index = std::ceil(in_x);
                const float x_lerp = in_x - left_x_index;

                for (int d = 0; d < depth; ++d)
                {   
                    const float *pimage = image_data + b_in * image_elements + d * image_channel_elements;

                    const float top_left = pimage[top_y_index * image_width + left_x_index];
                    const float top_right = pimage[top_y_index * image_width + right_x_index];
                    const float bottom_left = pimage[bottom_y_index * image_width + left_x_index];
                    const float bottom_right = pimage[bottom_y_index * image_width + right_x_index];
                    
                    const float top = top_left + (top_right - top_left) * x_lerp;
                    const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
                        
                    corps_data[crop_elements * b + channel_elements * d + y * crop_width + x] = top + (bottom - top) * y_lerp;
                }
            }   // end for x
        }   // end for y
    }   // end for b

}


void align_roi_pool_forward(
    const py::array_t<float>& image,  //THFloatTensor * image,
    const py::array_t<float>& boxes,  //THFloatTensor * boxes,            // [y1, x1, y2, x2]
    const py::array_t<int>& box_inds, //THIntTensor * box_index,          // range in [0, batch_size)
    const float extrapolation_value,
    const int crop_height,
    const int crop_width,
    const float spatial_scale,
    py::array_t<float> & crops        //THFloatTensor * crops
) {
    /*
    const int batch_size = image->size[0];
    const int depth = image->size[1];
    const int image_height = image->size[2];
    const int image_width = image->size[3];

    const int num_boxes = boxes->size[0];
    */
  	py::buffer_info image_buf = image.request();
	py::buffer_info boxes_buf = boxes.request();
    py::buffer_info box_inds_buf = box_inds.request();
    py::buffer_info crops_buf = crops.request();

	const int batch_size = image_buf.shape[0];
    const int depth = image_buf.shape[1];
    const int image_height = image_buf.shape[2];
    const int image_width = image_buf.shape[3];

	const int num_boxes = boxes_buf.shape[0];

    // init output space
    //THFloatTensor_resize4d(crops, num_boxes, depth, crop_height, crop_width);
    //THFloatTensor_zero(crops);
	//apply for new memory
	//auto crops = py::array_t<float>(num_boxes*depth*crop_height*crop_width);
	//reshape to 4d array
	//crops.resize({num_boxes, depth, crop_height, crop_width});

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
    py::array_t<float> & grads_image    //THFloatTensor * grads_image // resize to [bsize, c, hc, wc]
)
{   
    // shape
    /*
    const int batch_size = grads_image->size[0];
    const int depth = grads_image->size[1];
    const int image_height = grads_image->size[2];
    const int image_width = grads_image->size[3];

    const int num_boxes = grads->size[0];
    const int crop_height = grads->size[2];
    const int crop_width = grads->size[3];
    */
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

    // init output space
    // THFloatTensor_zero(grads_image);

    // data pointer
    const float * grads_data = (const float*)grads_buf.ptr;
    const float * boxes_data = (const float*)boxes_buf.ptr;
    const int * box_index_data = (const int*)box_inds_buf.ptr;
    float * grads_image_data = (float*)grads_image_buf.ptr;

    for (int b = 0; b < num_boxes; ++b) {
        const float * box = boxes_data + b * 4;
        const float y1 = box[0];
        const float x1 = box[1];
        const float y2 = box[2];
        const float x2 = box[3];

        const int b_in = box_index_data[b];
        if (b_in < 0 || b_in >= batch_size) {
            char msg[100];
            sprintf(msg,"Error: batch_index %d out of range [0, %d)\n", b_in, batch_size);
            throw std::runtime_error(msg);
       }

        const float height_scale =
            (crop_height > 1) ? (y2 - y1) * spatial_scale / (crop_height - 1)
                              : 0;
        const float width_scale =
            (crop_width > 1) ? (x2 - x1) * spatial_scale / (crop_width - 1)
                             : 0;

        for (int y = 0; y < crop_height; ++y)
        {
            const float in_y = (crop_height > 1)
                                   ? y1 * spatial_scale + y * height_scale
                                   : 0.5f * (y1 + y2) * spatial_scale;
            if (in_y < 0 || in_y > image_height - 1)
            {
                continue;
            }
            const int top_y_index = floorf(in_y);
            const int bottom_y_index = ceilf(in_y);
            const float y_lerp = in_y - top_y_index;

            for (int x = 0; x < crop_width; ++x)
            {
                const float in_x = (crop_width > 1)
                                       ? x1 * spatial_scale + x * width_scale
                                       : 0.5f * (x1 + x2) * spatial_scale;
                if (in_x < 0 || in_x > image_width - 1)
                {
                    continue;
                }
                const int left_x_index = floorf(in_x);
                const int right_x_index = ceilf(in_x);
                const float x_lerp = in_x - left_x_index;

                for (int d = 0; d < depth; ++d)
                {   
                    float *pimage = grads_image_data + b_in * image_elements + d * image_channel_elements;
                    const float grad_val = grads_data[crop_elements * b + channel_elements * d + y * crop_width + x];

                    const float dtop = (1 - y_lerp) * grad_val;
                    pimage[top_y_index * image_width + left_x_index] += (1 - x_lerp) * dtop;
                    pimage[top_y_index * image_width + right_x_index] += x_lerp * dtop;

                    const float dbottom = y_lerp * grad_val;
                    pimage[bottom_y_index * image_width + left_x_index] += (1 - x_lerp) * dbottom;
                    pimage[bottom_y_index * image_width + right_x_index] += x_lerp * dbottom;
                }   // end d
            }   // end x
        }   // end y
    }   // end b
}


PYBIND11_MODULE(align_roi_pool_mh, m) {
	m.doc() = "do align roi pooling forward backward with pybind11";
	m.attr("__version__") = "0.0.1";
	m.def("add", &add, "for test");
	m.def("align_roi_pool_forward", &align_roi_pool_forward, "A function does align roi forward", \
        py::arg("image"), py::arg("boxes"), py::arg("box_inds"), py::arg("extrapolation_value"), \
        py::arg("crop_height"), py::arg("crop_width"), py::arg("spatial_scale"), py::arg("crops"));
	m.def("align_roi_pool_backward", &align_roi_pool_backward, "A function does align roi backward", \
		py::arg("grads"), py::arg("boxes"), py::arg("box_inds"), py::arg("spatial_scale"), py::arg("grads_image"));
}