// Copyright (c) 2020 Fetullah Atas, Norwegian University of Life Sciences
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

#include "fidnet/resnet.hpp"


resnet::BasicBlock::BasicBlock(
  int inplanes, int planes, int stride = 1, int downsample = 0, int groups = 1,
  int base_width = 64, int dilation = 1, int if_BN = 0)
: conv1_(conv3x3(inplanes, planes, stride)),
  batch_norm1_(torch::nn::BatchNorm2dOptions(planes)),
  conv2_(conv3x3(planes, planes)),
  batch_norm2_(torch::nn::BatchNorm2dOptions(planes)),
  relu_(torch::nn::LeakyReLU()),
  stride_(stride),
  downsample_(downsample),
  if_BN_(if_BN)
{
  std::cout << "Constructing a BasicBlock object." << std::endl;
}

resnet::BasicBlock::~BasicBlock()
{
}

at::Tensor resnet::BasicBlock::forward(at::Tensor x)
{
  auto identity = x;
  auto out = conv1_(x);
  if (if_BN_) {
    out = batch_norm1_(out);
  }
  out = conv2_(out);
  if (if_BN_) {
    out = batch_norm2_(out);
  }
  if (!downsample_) {
    //identity = self.downsample(x)
    std::logic_error("BasicBlock::forward !downsample_ not yet implemented");
  }
  out += identity;
  out = relu_(out);
  return out;
}
