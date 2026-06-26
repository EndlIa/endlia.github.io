---
title: "CSAPP 练习：在 -O0 下优化 arraymul"
date: 2026-06-27T00:00:00+08:00
description: "分析一个逐元素数组乘法函数在 -O0、AVX2 条件下的优化策略。"
tags: ["CSAPP", "性能优化", "SIMD", "C"]
draft: false
---

这道题要求优化一个很简单的函数：把两个 `int` 数组逐元素相乘，结果写入第三个数组。

baseline 如下：

```c
void arraymul(vec * v1, vec * v2, vec *v3){
    for (size_t i = 0; i < v1->len; i++) {
        v3->data[i] = v1->data[i] * v2->data[i];
    }
}
```

看起来它已经足够简单，但题目里有一个很关键的评测条件：

```text
-w -mavx2 -O0 -pthread
```

也就是说，编译器不会帮我们做高级优化。平时在 `-O2` 或 `-O3` 下可能自动完成的事情，在这里基本都要自己写出来。

## baseline 慢在哪里

baseline 每轮循环都要做这些事情：

1. 读取 `v1->len`
2. 读取 `v1->data`
3. 读取 `v2->data`
4. 读取 `v3->data`
5. 计算数组下标
6. 做一次乘法和一次写回

在 `-O0` 下，编译器通常不会把 `v1->len`、`v1->data` 这些字段缓存到寄存器里，也不会主动展开循环或向量化。因此这个循环虽然逻辑简单，但每次迭代都有不少额外开销。

第一步优化应该是把循环不变量取出来：

```c
size_t len = v1->len;
int *a = v1->data;
int *b = v2->data;
int *c = v3->data;
```

这样循环体里就不需要反复访问结构体字段。

## 为什么 SIMD 是核心策略

题目允许使用 `student.h`，里面已经包含了：

```c
#include <immintrin.h>
#include <x86intrin.h>
```

同时编译选项有：

```text
-mavx2
```

这意味着我们可以直接使用 AVX2 intrinsic。

`int` 是 32 位，AVX2 的 `__m256i` 是 256 位，所以一次可以处理 8 个 `int`：

```c
__m256i va = _mm256_loadu_si256((__m256i *)(a + i));
__m256i vb = _mm256_loadu_si256((__m256i *)(b + i));
__m256i vc = _mm256_mullo_epi32(va, vb);
_mm256_storeu_si256((__m256i *)(c + i), vc);
```

这里使用 `loadu/storeu`，因为题目没有保证数组 32 字节对齐。未对齐加载在现代 x86 上通常还能接受，而且比为了对齐写复杂分支更稳。

## 是否需要多线程

题目编译选项里有 `-pthread`，理论上可以多线程。但是这道题不一定适合一上来就用线程。

原因是：

- 如果数组较小，创建线程的开销可能比计算本身还大。
- 评测可能会调用很多次函数，频繁创建线程会非常亏。
- 题目没有说明运行环境有几个核心，也没有说明数据规模。
- `arraymul` 是纯内存流式访问，很多时候瓶颈会很快变成内存带宽。

所以更稳的策略是：先写一个高质量的单线程 AVX2 版本。它实现简单、正确性风险低，对多数规模都有效。

如果已知评测数组极大，再考虑固定线程池或分块多线程。但在这个题目给出的接口中，每次调用 `arraymul` 都临时建线程，未必划算。

## 推荐实现

一个比较稳的实现如下：

```c
#include "student.h"

typedef struct{
    size_t len;
    int *data;
} vec;

void arraymul(vec * v1, vec * v2, vec *v3){
    size_t len = v1->len;
    int *a = v1->data;
    int *b = v2->data;
    int *c = v3->data;

    size_t i = 0;
    size_t limit = len & ~(size_t)7;

    for (; i < limit; i += 8) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));
        __m256i vc = _mm256_mullo_epi32(va, vb);
        _mm256_storeu_si256((__m256i *)(c + i), vc);
    }

    for (; i < len; i++) {
        c[i] = a[i] * b[i];
    }
}
```

这段代码的思路很直接：

- 先把结构体字段缓存到局部变量。
- 主循环每次处理 8 个 `int`。
- 剩下不足 8 个的尾部元素用普通循环处理。
- 不依赖对齐假设，因此不会因为数据地址不对齐而出错。

## 还能不能继续优化

可以考虑循环展开，例如一次处理 16 或 32 个 `int`：

```c
for (; i + 15 < len; i += 16) {
    __m256i a0 = _mm256_loadu_si256((const __m256i *)(a + i));
    __m256i b0 = _mm256_loadu_si256((const __m256i *)(b + i));
    __m256i c0 = _mm256_mullo_epi32(a0, b0);
    _mm256_storeu_si256((__m256i *)(c + i), c0);

    __m256i a1 = _mm256_loadu_si256((const __m256i *)(a + i + 8));
    __m256i b1 = _mm256_loadu_si256((const __m256i *)(b + i + 8));
    __m256i c1 = _mm256_mullo_epi32(a1, b1);
    _mm256_storeu_si256((__m256i *)(c + i + 8), c1);
}
```

循环展开可以减少分支判断和循环变量更新的次数，也能给 CPU 更多独立指令。但展开太多会让代码变长，在 `-O0` 下也可能带来寄存器分配和指令缓存方面的问题。

这题比较合理的取舍是展开到 16 个或 32 个 `int`，不要无限展开。

## 一个展开版

如果希望再激进一点，可以写成 4 路 AVX2 展开，一次处理 32 个 `int`：

```c
#include "student.h"

typedef struct{
    size_t len;
    int *data;
} vec;

void arraymul(vec * v1, vec * v2, vec *v3){
    size_t len = v1->len;
    int *a = v1->data;
    int *b = v2->data;
    int *c = v3->data;

    size_t i = 0;
    size_t limit32 = len & ~(size_t)31;

    for (; i < limit32; i += 32) {
        __m256i a0 = _mm256_loadu_si256((const __m256i *)(a + i));
        __m256i b0 = _mm256_loadu_si256((const __m256i *)(b + i));
        __m256i r0 = _mm256_mullo_epi32(a0, b0);
        _mm256_storeu_si256((__m256i *)(c + i), r0);

        __m256i a1 = _mm256_loadu_si256((const __m256i *)(a + i + 8));
        __m256i b1 = _mm256_loadu_si256((const __m256i *)(b + i + 8));
        __m256i r1 = _mm256_mullo_epi32(a1, b1);
        _mm256_storeu_si256((__m256i *)(c + i + 8), r1);

        __m256i a2 = _mm256_loadu_si256((const __m256i *)(a + i + 16));
        __m256i b2 = _mm256_loadu_si256((const __m256i *)(b + i + 16));
        __m256i r2 = _mm256_mullo_epi32(a2, b2);
        _mm256_storeu_si256((__m256i *)(c + i + 16), r2);

        __m256i a3 = _mm256_loadu_si256((const __m256i *)(a + i + 24));
        __m256i b3 = _mm256_loadu_si256((const __m256i *)(b + i + 24));
        __m256i r3 = _mm256_mullo_epi32(a3, b3);
        _mm256_storeu_si256((__m256i *)(c + i + 24), r3);
    }

    size_t limit8 = len & ~(size_t)7;

    for (; i < limit8; i += 8) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));
        __m256i vc = _mm256_mullo_epi32(va, vb);
        _mm256_storeu_si256((__m256i *)(c + i), vc);
    }

    for (; i < len; i++) {
        c[i] = a[i] * b[i];
    }
}
```

这个版本仍然只包含题目允许的头文件：

```c
#include "student.h"
```

没有额外宏定义，也没有依赖错误处理。

## 小结

这道题的核心不是“乘法怎么写”，而是理解评测条件。

在 `-O0` 下，编译器不会主动帮我们做很多优化，所以手写优化非常有价值。最关键的几步是：

1. 把 `len` 和 `data` 指针缓存到局部变量。
2. 使用 AVX2 一次处理 8 个 `int`。
3. 做适度循环展开，降低循环控制开销。
4. 用普通标量循环处理尾部。
5. 不轻易上多线程，除非确认数据规模足够大且线程开销可控。

对这类简单数组计算来说，单线程 AVX2 往往已经能接近内存带宽上限。优化的重点，是减少 `-O0` 下的额外开销，并让每次循环尽可能多地完成有效工作。
