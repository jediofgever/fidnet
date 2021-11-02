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

resnet::ResNet_34_point::ResNet_34_point(
  BasicBlock block, std::vector<int> layers,
  int if_BN, bool if_remission, bool if_range, bool with_normal,
  bool zero_init_residual = false,
  int norm_layer = 0, int groups = 1, int width_per_group = 64)
: if_BN_(if_BN),
  if_remission_(if_remission),
  if_range_(if_range),
  with_normal_(with_normal),
  in_planes_(512),
  dilation_(1),
  groups_(groups),
  base_width_(width_per_group),
  norm_layer_(norm_layer)
{

  if (!if_remission_ && !if_range_ && !with_normal_) {
    conv1_ = std::make_shared<torch::nn::Conv2d>(
      torch::nn::Conv2dOptions(3, 64, 1).stride(1).padding(0).bias(true));
    bn0_ = std::make_shared<torch::nn::BatchNorm2d>(torch::nn::BatchNorm2dOptions(64));
    relu0_ = std::make_shared<torch::nn::LeakyReLU>();
  }

  if (if_remission_ && !if_range_ && !with_normal_) {
    conv1_ = std::make_shared<torch::nn::Conv2d>(
      torch::nn::Conv2dOptions(4, 64, 1).stride(1).padding(0).bias(true));
    bn0_ = std::make_shared<torch::nn::BatchNorm2d>(torch::nn::BatchNorm2dOptions(64));
    relu0_ = std::make_shared<torch::nn::LeakyReLU>();
  }

  if (if_remission_ && if_range_ && !with_normal_) {
    conv1_ = std::make_shared<torch::nn::Conv2d>(
      torch::nn::Conv2dOptions(5, 64, 1).stride(1).padding(0).bias(true));
    bn0_ = std::make_shared<torch::nn::BatchNorm2d>(torch::nn::BatchNorm2dOptions(64));
    relu0_ = std::make_shared<torch::nn::LeakyReLU>();
  }

  if (if_remission_ && if_range_ && with_normal_) {
    conv1_ = std::make_shared<torch::nn::Conv2d>(
      torch::nn::Conv2dOptions(8, 64, 1).stride(1).padding(0).bias(true));
    bn0_ = std::make_shared<torch::nn::BatchNorm2d>(torch::nn::BatchNorm2dOptions(64));
    relu0_ = std::make_shared<torch::nn::LeakyReLU>();
  }


  conv2_ = std::make_shared<torch::nn::Conv2d>(
    torch::nn::Conv2dOptions(64, 128, 1).stride(1).padding(0).bias(true));
  bn_ = std::make_shared<torch::nn::BatchNorm2d>(torch::nn::BatchNorm2dOptions(128));
  relu_ = std::make_shared<torch::nn::LeakyReLU>();

  conv3_ = std::make_shared<torch::nn::Conv2d>(
    torch::nn::Conv2dOptions(128, 256, 1).stride(1).padding(0).bias(true));
  bn1_ = std::make_shared<torch::nn::BatchNorm2d>(torch::nn::BatchNorm2dOptions(256));
  relu1_ = std::make_shared<torch::nn::LeakyReLU>();

  conv4_ = std::make_shared<torch::nn::Conv2d>(
    torch::nn::Conv2dOptions(256, 512, 1).stride(1).padding(0).bias(true));
  bn2_ = std::make_shared<torch::nn::BatchNorm2d>(torch::nn::BatchNorm2dOptions(512));
  relu2_ = std::make_shared<torch::nn::LeakyReLU>();

  layer1_ = make_layer(block, 128, layers[0], 1 /*stride*/, false /*dilate*/);
  layer2_ = make_layer(block, 128, layers[1], 2 /*stride*/, false /*dilate*/);
  layer3_ = make_layer(block, 128, layers[2], 2 /*stride*/, false /*dilate*/);
  layer4_ = make_layer(block, 128, layers[3], 2 /*stride*/, false /*dilate*/);

  for (auto i : this->modules()) {
    if (instanceof<torch::nn::Conv2d>(&(*i))) {
      torch::nn::init::kaiming_normal_(
        i->as<torch::nn::Conv2d>()->weight, 0.0, torch::kFanOut, torch::kLeakyReLU);

    }

    if (instanceof<torch::nn::BatchNorm2d>(&(*i)) || instanceof<torch::nn::GroupNorm>(&(*i))) {
      for (auto && j : i->parameters()) {
        torch::nn::init::constant_(i->as<torch::nn::GroupNorm>()->weight, 1);
        torch::nn::init::constant_(i->as<torch::nn::BatchNorm2d>()->bias, 0);
      }
    }

  }

  if (zero_init_residual) {
    for (auto i : this->modules()) {
      if (instanceof<BasicBlock>(&(*i))) {
        torch::nn::init::constant_(i->as<BasicBlock>()->batch_norm2_->weight, 0);
      }
    }

  }


  std::cout << "Constructing a ResNet_34_point object." << std::endl;
}

resnet::ResNet_34_point::~ResNet_34_point()
{
}

at::Tensor resnet::ResNet_34_point::forward(at::Tensor x)
{

}

torch::nn::Sequential resnet::ResNet_34_point::make_layer(
  BasicBlock block, int planes, int blocks, int stride = 1,
  bool dilate = false)
{
  auto norm_layer = norm_layer_;
  torch::nn::Sequential downsample;
  auto previous_dilation = dilation_;

  if (dilate) {
    dilation_ *= stride;
    stride = 1;
  }

  if (stride != 1 && in_planes_ != planes * block.expansion_) {
    if (if_BN_) {
      downsample = torch::nn::Sequential(
        conv1x1(in_planes_, planes * block.expansion_, stride),
        torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(planes * block.expansion_)));
    } else {
      torch::nn::Sequential(
        conv1x1(in_planes_, planes * block.expansion_, stride));
    }
  }

  torch::nn::Sequential layers;
  layers->push_back(
    BasicBlock(
      in_planes_, planes, 1 /*stride*/, 0 /*downsample*/,
      groups_, base_width_, previous_dilation, if_BN_));
  in_planes_ = planes * block.expansion_;

  for (size_t i = 1; i < blocks; i++) {
    layers->push_back(
      BasicBlock(
        in_planes_, planes, 1 /*stride*/, 0 /*downsample*/,
        groups_, base_width_, dilation_, if_BN_));
  }
  return layers;
}
