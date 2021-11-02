// Copyright (c) 2021 Fetullah Atas, Norwegian University of Life Sciences
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef RESNET
#define RESNET

#include "torch/torch.h"

namespace resnet
{

template<typename Base, typename T>
inline bool instanceof(const T *)
{
  return is_base_of<Base, T>::value;
}

torch::nn::Conv1d conv1x1(
  int in_planes, int out_planes, int stride = 1)
{
  return torch::nn::Conv1d(
    torch::nn::Conv1dOptions(in_planes, out_planes, 1)
    .stride(stride)
    .bias(false));
}

torch::nn::Conv2d conv3x3(
  int in_planes, int out_planes, int stride = 1,
  int groups = 1, int dilation = 1)
{
  return torch::nn::Conv2d(
    torch::nn::Conv2dOptions(in_planes, out_planes, 3)
    .stride(stride)
    .groups(groups)
    .dilation(dilation)
    .bias(false));
}

class BasicBlock : public torch::nn::Module
{

private:
public:
  BasicBlock(
    int inplanes, int planes, int stride = 1, int downsample = 0, int groups = 1,
    int base_width = 64, int dilation = 1, int if_BN = 0);
  ~BasicBlock();
  at::Tensor forward(at::Tensor x);

  int expansion_ = 1;
  int stride_;
  bool downsample_;
  bool if_BN_;
  torch::nn::Conv2d conv1_, conv2_;
  torch::nn::BatchNorm2d batch_norm1_, batch_norm2_;
  torch::nn::LeakyReLU relu_;

};


class ResNet_34_point : public torch::nn::Module
{

private:
  bool if_BN_;
  bool if_remission_;
  bool if_range_;
  bool with_normal_;
  int groups_;
  int base_width_;
  int in_planes_ = 512;
  int dilation_ = 1;
  int norm_layer_;

  std::shared_ptr<torch::nn::Conv2d> conv1_, conv2_, conv3_, conv4_;
  std::shared_ptr<torch::nn::BatchNorm2d> bn0_, bn_, bn1_, bn2_;
  std::shared_ptr<torch::nn::LeakyReLU> relu0_, relu_, relu1_, relu2_;

  torch::nn::Sequential layer1_;
  torch::nn::Sequential layer2_;
  torch::nn::Sequential layer3_;
  torch::nn::Sequential layer4_;

public:
  ResNet_34_point(
    BasicBlock block, std::vector<int> layers,
    int if_BN, bool if_remission, bool if_range, bool with_normal,
    bool zero_init_residual = false,
    int norm_layer = 0, int groups = 1, int width_per_group = 64);
  ~ResNet_34_point();
  at::Tensor forward(at::Tensor x);

  torch::nn::Sequential make_layer(
    BasicBlock block, int planes, int blocks, int stride = 1,
    bool dilate = false);
};

}  // namespace resnet


#endif
