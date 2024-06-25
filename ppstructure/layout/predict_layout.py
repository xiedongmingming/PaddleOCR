# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))

sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../..")))

os.environ["FLAGS_allocator_strategy"] = "auto_growth"

import cv2
import numpy as np
import time

import tools.infer.utility as utility

from ppocr.data import create_operators, transform
from ppocr.postprocess import build_post_process
from ppocr.utils.logging import get_logger
from ppocr.utils.utility import get_image_file_list, check_and_read
from ppstructure.utility import parse_args
from picodet_postprocess import PicoDetPostProcess

logger = get_logger()


class LayoutPredictor(object):

    def __init__(self, args):

        pre_process_list = [
            {"Resize": {"size": [800, 608]}},
            {
                "NormalizeImage": {
                    "std": [0.229, 0.224, 0.225],
                    "mean": [0.485, 0.456, 0.406],
                    "scale": "1./255.",
                    "order": "hwc",
                }
            },
            {"ToCHWImage": None},
            {"KeepKeys": {"keep_keys": ["image"]}},
        ]

        postprocess_params = {
            "name": "PicoDetPostProcess",
            "layout_dict_path": args.layout_dict_path,
            "score_threshold": args.layout_score_threshold,
            "nms_threshold": args.layout_nms_threshold,
        }

        self.preprocess_op = create_operators(pre_process_list)
        self.postprocess_op = build_post_process(postprocess_params)

        (
            self.predictor,  # <paddle.fluid.libpaddle.PaddleInferPredictor object at 0x000001F4BC762A70>
            self.input_tensor,  # <paddle.fluid.libpaddle.PaddleInferTensor object at 0x000001F4B7CB8330>
            self.output_tensors, # 8个：[<paddle.fluid.libpaddle.PaddleInferTensor object at 0x000001F4AF4FBBF0>, <paddle.fluid.libpaddle.PaddleInferTensor object at 0x000001F4AF4FBA70>, <paddle.fluid.libpaddle.PaddleInferTensor object at 0x000001F4AF4FA330>, <paddle.fluid.libpaddle.PaddleInferTensor object at 0x000001F4AF4F94F0>, <paddle.fluid.libpaddle.PaddleInferTensor object at 0x000001F4AF4FB5F0>, <paddle.fluid.libpaddle.PaddleInferTensor object at 0x000001F4AF4FB670>, <paddle.fluid.libpaddle.PaddleInferTensor object at 0x000001F4AF4FA7F0>, <paddle.fluid.libpaddle.PaddleInferTensor object at 0x000001F4AF4FBCB0>]
            self.config,  # <paddle.fluid.libpaddle.AnalysisConfig object at 0x000001F4ADDCE030>
        ) = utility.create_predictor(args, "layout", logger)

        self.use_onnx = args.use_onnx

    def __call__(self, img): # 步骤3.4.1：执行推理（父类实现）-- 布局推理 对外接口 -- 真正执行功能的入口

        ori_im = img.copy()

        data = {"image": img}

        data = transform(data, self.preprocess_op) # 步骤3.4.2：执行推理（父类实现）-- 布局推理前置处理 list<ndarray: (3, 800, 608)>

        img = data[0] # ndarray: (3, 800, 608)

        if img is None:

            return None, 0

        img = np.expand_dims(img, axis=0) # ndarray: (1, 3, 800, 608)

        img = img.copy()

        preds, elapse = 0, 1

        starttime = time.time()

        np_score_list, np_boxes_list = [], []

        if self.use_onnx:

            input_dict = {}

            input_dict[self.input_tensor.name] = img

            outputs = self.predictor.run(self.output_tensors, input_dict)

            num_outs = int(len(outputs) / 2)

            for out_idx in range(num_outs):

                np_score_list.append(outputs[out_idx])
                np_boxes_list.append(outputs[out_idx + num_outs])

        else:

            self.input_tensor.copy_from_cpu(img) # <paddle.fluid.libpaddle.PaddleInferTensor object at 0x0000023E7385F630>

            self.predictor.run() # 步骤3.4.3：执行推理（父类实现）-- 布局推理执行RUN

            output_names = self.predictor.get_output_names() # ['transpose_0.tmp_0', 'transpose_2.tmp_0', 'transpose_4.tmp_0', 'transpose_6.tmp_0', 'transpose_1.tmp_0', 'transpose_3.tmp_0', 'transpose_5.tmp_0', 'transpose_7.tmp_0']

            num_outs = int(len(output_names) / 2)

            for out_idx in range(num_outs):

                np_score_list.append(
                    self.predictor.get_output_handle(
                        output_names[out_idx]
                    ).copy_to_cpu()
                )
                np_boxes_list.append(
                    self.predictor.get_output_handle(
                        output_names[out_idx + num_outs]
                    ).copy_to_cpu()
                )

        preds = dict(boxes=np_score_list, boxes_num=np_boxes_list)

        post_preds = self.postprocess_op(ori_im, img, preds)

        elapse = time.time() - starttime

        return post_preds, elapse


def main(args):

    image_file_list = get_image_file_list(args.image_dir)

    layout_predictor = LayoutPredictor(args)

    count = 0

    total_time = 0

    repeats = 50

    for image_file in image_file_list:

        img, flag, _ = check_and_read(image_file)

        if not flag:

            img = cv2.imread(image_file)

        if img is None:

            logger.info("error in loading image:{}".format(image_file))

            continue

        layout_res, elapse = layout_predictor(img)

        logger.info("result: {}".format(layout_res))

        if count > 0:

            total_time += elapse

        count += 1

        logger.info("Predict time of {}: {}".format(image_file, elapse))


if __name__ == "__main__":

    main(parse_args())
