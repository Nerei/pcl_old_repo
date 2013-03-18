/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2011, Willow Garage, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include "device.hpp"
#include "texture_binder.hpp"


namespace pcl
{
  namespace device
  {
    texture<ushort, 2> dprev_tex;
    texture<Normal, 2> nprev_tex;
    texture<Point,  2> vprev_tex;

    struct ComputeIcpHelper::Policy
    {
      enum
      {
        CTA_SIZE_X = 32,
        CTA_SIZE_Y = 8,
        CTA_SIZE = CTA_SIZE_X * CTA_SIZE_Y,
        
        B = 6, COLS = 6, ROWS = 6, DIAG = 6,
        UPPER_DIAG_MAT = (COLS * ROWS - DIAG) / 2 + DIAG,
        TOTAL = UPPER_DIAG_MAT + B,

        FINAL_REDUCE_CTA_SIZE = 256,
        FINAL_REDUCE_STRIDE = FINAL_REDUCE_CTA_SIZE
      };
    };

    __device__ __forceinline__
    float2 ComputeIcpHelper::proj(const float3& p) const
    {
       float2 coo;
       coo.x = __fmaf_rn(f.x, __fdividef(p.x, p.z), c.x);
       coo.y = __fmaf_rn(f.y, __fdividef(p.y, p.z), c.y);
       return coo;
    }

    __device__ __forceinline__
    float3 ComputeIcpHelper::reproj(float u, float v, float z)  const
    {
      float x = z * (u - c.x) * finv.x;
      float y = z * (v - c.y) * finv.y;
      return make_float3(x, y, z);
    }
#if defined USE_DEPTH
    __device__ __forceinline__
    int ComputeIcpHelper::find_coresp(int x, int y, float3& nd, float3& d, float3& s) const
    {
      int src_z = dcurr(y, x);
      if (src_z == 0)
          return 40;

      s = reproj(x, y, src_z * 0.001f);
      s = R * s + t;

      float2 coo = proj(s);
      if (s.z <= 0 || coo.x < 0 || coo.y < 0 || coo.x >= cols || coo.y >= rows)
        return 80;

      int dst_z = tex2D(dprev_tex, coo.x, coo.y);
      if (dst_z == 0)
          return 120;

      d = reproj(coo.x, coo.y, dst_z * 0.001f);
      
      float dist2 = norm_sqr(s - d);
      if (dist2 > dist2_thres)
        return 160;

      float3 ns = R * tr(ncurr(y, x));
      nd = tr(tex2D(nprev_tex, coo.x, coo.y));
      
      float cosine = fabs(dot(ns, nd));
      if (cosine < min_cosine)
        return 200;
      return 0;
    }
#else
     __device__ __forceinline__
    int ComputeIcpHelper::find_coresp(int x, int y, float3& nd, float3& d, float3& s) const
    {
      s = tr(vcurr(y, x));
      if (isnan(s.x))
          return 40;

      s = R * s + t;

      float2 coo = proj(s);
      if (s.z <= 0 || coo.x < 0 || coo.y < 0 || coo.x >= cols || coo.y >= rows)
        return 80;

      d = tr(tex2D(vprev_tex, coo.x, coo.y));
      if (isnan(d.x))
          return 120;

      float dist2 = norm_sqr(s - d);
      if (dist2 > dist2_thres)
        return 160;

      float3 ns = R * tr(ncurr(y, x));
      nd = tr(tex2D(nprev_tex, coo.x, coo.y));
      
      float cosine = fabs(dot(ns, nd));
      if (cosine < min_cosine)
        return 200;
      return 0;
    }
#endif

     __device__ __forceinline__
    void ComputeIcpHelper::partial_reduce(const float row[7], PtrStep<float>& partial_buf) const
    {
      volatile __shared__ float smem[Policy::CTA_SIZE];
      int tid = Block::flattenedThreadId ();

      float  *pos = partial_buf.data + blockIdx.x + gridDim.x * blockIdx.y;
      size_t step = partial_buf.step / sizeof(float);

      for (int i = 0; i < 6; ++i)        //rows
      {
        #pragma unroll
        for (int j = i; j < 7; ++j)      // cols + b
        {
          __syncthreads ();
          smem[tid] = row[i] * row[j];
          __syncthreads ();

          Block::reduce<Policy::CTA_SIZE>(smem, plus ());

          if (tid == 0)
          {
              *pos = smem[0];
              pos += step;
          }
        }
      }
    }

    __global__ void icp_helper_kernel(const ComputeIcpHelper helper, PtrStep<float> partial_buf)
    { 
      int x = threadIdx.x + blockIdx.x * ComputeIcpHelper::Policy::CTA_SIZE_X;
      int y = threadIdx.y + blockIdx.y * ComputeIcpHelper::Policy::CTA_SIZE_Y;
      
      float3 n, d, s;
      int filtered = (x < helper.cols && y < helper.rows) ? helper.find_coresp (x, y, n, d, s) : 1;
      //if (x < helper.cols && y < helper.rows) mask(y, x) = filtered;
      
      float row[7];

      if (!filtered)
      {
        *(float3*)&row[0] = cross (s, n);
        *(float3*)&row[3] = n;
        row[6] = dot (n, d - s);
      }
      else
        row[0] = row[1] = row[2] = row[3] = row[4] = row[5] = row[6] = 0.f;

      helper.partial_reduce(row, partial_buf);
    }

    __global__ void icp_final_reduce_kernel(const PtrStep<float> partial_buf, const int length, float* final_buf)
    {
      const float *beg = partial_buf.ptr(blockIdx.x);
      const float *end = beg + length;

      int tid = threadIdx.x;

      float sum = 0.f;
      for (const float *t = beg + tid; t < end; t += ComputeIcpHelper::Policy::FINAL_REDUCE_STRIDE)
          sum += *t;

      __shared__ float smem[ComputeIcpHelper::Policy::FINAL_REDUCE_CTA_SIZE];

      smem[tid] = sum;

      __syncthreads();

      Block::reduce<ComputeIcpHelper::Policy::FINAL_REDUCE_CTA_SIZE>(smem, plus());

      if (tid == 0)
        final_buf[blockIdx.x] = smem[0];
    };
  }
}

void pcl::device::ComputeIcpHelper::operator()(const Depth& dprev, const Normals& nprev, DeviceArray2D<float>& buffer, float* data, cudaStream_t s)
{
  dprev_tex.filterMode = cudaFilterModePoint;
  nprev_tex.filterMode = cudaFilterModePoint;
  TextureBinder dprev_binder(dprev, dprev_tex);
  TextureBinder nprev_binder(nprev, nprev_tex);

  dim3 block(Policy::CTA_SIZE_X, Policy::CTA_SIZE_Y);
  dim3 grid(divUp ((int)cols, block.x), divUp ((int)rows, block.y));

  int partials_count = (int)(grid.x * grid.y);
  allocate_buffer(buffer, partials_count);

  icp_helper_kernel<<<grid, block, 0, s>>>(*this, buffer);
  cudaSafeCall ( cudaGetLastError () );

  int b = Policy::FINAL_REDUCE_CTA_SIZE;
  int g = Policy::TOTAL;
  icp_final_reduce_kernel<<<g, b, 0, s>>>(buffer, partials_count, buffer.ptr(Policy::TOTAL));
  cudaSafeCall ( cudaGetLastError () );

  cudaSafeCall ( cudaMemcpyAsync(data, buffer.ptr(Policy::TOTAL), Policy::TOTAL * sizeof(float), cudaMemcpyDeviceToHost, s) );
  cudaSafeCall ( cudaGetLastError () );
}

void pcl::device::ComputeIcpHelper::operator()(const Points& vprev, const Normals& nprev, DeviceArray2D<float>& buffer, float* data, cudaStream_t s)
{
  dprev_tex.filterMode = cudaFilterModePoint;
  nprev_tex.filterMode = cudaFilterModePoint;
  TextureBinder vprev_binder(vprev, vprev_tex);
  TextureBinder nprev_binder(nprev, nprev_tex);

  dim3 block(Policy::CTA_SIZE_X, Policy::CTA_SIZE_Y);
  dim3 grid(divUp ((int)cols, block.x), divUp ((int)rows, block.y));

  int partials_count = (int)(grid.x * grid.y);
  allocate_buffer(buffer, partials_count);

  icp_helper_kernel<<<grid, block, 0, s>>>(*this, buffer);
  cudaSafeCall ( cudaGetLastError () );

  int b = Policy::FINAL_REDUCE_CTA_SIZE;
  int g = Policy::TOTAL;
  icp_final_reduce_kernel<<<g, b, 0, s>>>(buffer, partials_count, buffer.ptr(Policy::TOTAL));
  cudaSafeCall ( cudaGetLastError () );

  cudaSafeCall ( cudaMemcpyAsync(data, buffer.ptr(Policy::TOTAL), Policy::TOTAL * sizeof(float), cudaMemcpyDeviceToHost, s) );
  cudaSafeCall ( cudaGetLastError () );
}


void pcl::device::ComputeIcpHelper::allocate_buffer(DeviceArray2D<float>& buffer, int partials_count)
{ 
  if (partials_count < 0)
  {
    const int input_cols = 640;
    const int input_rows = 480;

    int gx = divUp (input_cols, Policy::CTA_SIZE_X);
    int gy = divUp (input_rows, Policy::CTA_SIZE_Y);

    partials_count = gx * gy;
  }

  int min_rows = Policy::TOTAL + 1;
  int min_cols = max(partials_count, Policy::TOTAL);

  if (buffer.rows() < min_rows || buffer.cols() < min_cols)
    buffer.create (min_rows, min_cols);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// ComputeIcpHelper::PageLockHelper

pcl::device::ComputeIcpHelper::PageLockHelper::PageLockHelper() : data(0)
{ cudaSafeCall( cudaMallocHost((void **)&data, Policy::TOTAL * sizeof(float)) );  }

pcl::device::ComputeIcpHelper::PageLockHelper::~PageLockHelper()
{   cudaSafeCall( cudaFreeHost(data) ); data = 0; }