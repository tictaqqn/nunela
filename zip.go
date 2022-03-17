package nunela

import (
	"sync"

	"github.com/vorduin/nune"
	"github.com/vorduin/slices"
)

// TryZip computes element-wise operation on two tensors.
func TryZip[T, U, V Number](a *nune.Tensor[T], b *nune.Tensor[U], f func(T, U) V) (*nune.Tensor[V], error) {
	if !slices.Equal(a.Shape(), b.Shape()) {
		return nil, NewErrDifferentShapesTwo(a, b)
	}
	out := nune.Zeros[V](a.Shape()...)
	handleZip(a, b, &out, f, configCPU(a.Numel()))
	return &out, nil
}

// TryZipAssign computes element-wise operation on two tensors and assigns the result to the first tensor.
func TryZipAssign[T, U Number](a *nune.Tensor[T], b *nune.Tensor[U], f func(T, U) T) error {
	if !slices.Equal(a.Shape(), b.Shape()) {
		return NewErrDifferentShapesTwo(a, b)
	}
	handleZip(a, b, a, f, configCPU(a.Numel()))
	return nil
}

// Zip computes element-wise operation on two tensors.
func Zip[T, U, V Number](a *nune.Tensor[T], b *nune.Tensor[U], f func(T, U) V) *nune.Tensor[V] {
	t, err := TryZip(a, b, f)
	if err != nil {
		panic(err)
	}
	return t
}

// ZipAssign computes element-wise operation on two tensors and assigns the result to the first tensor.
func ZipAssign[T, U Number](a *nune.Tensor[T], b *nune.Tensor[U], f func(T, U) T) {
	err := TryZipAssign(a, b, f)
	if err != nil {
		panic(err)
	}
}

// Add computes element-wise addition of two tensors.
func Add[T Number](a, b *nune.Tensor[T]) *nune.Tensor[T] {
	return Zip(a, b, func(x, y T) T { return x + y })
}

// Sub computes element-wise subtraction of two tensors.
func Sub[T Number](a, b *nune.Tensor[T]) *nune.Tensor[T] {
	return Zip(a, b, func(x, y T) T { return x - y })
}

// Mul computes element-wise multiplication of two tensors.
func Mul[T Number](a, b *nune.Tensor[T]) *nune.Tensor[T] {
	return Zip(a, b, func(x, y T) T { return x * y })
}

// Div computes element-wise division of two tensors.
func Div[T Number](a, b *nune.Tensor[T]) *nune.Tensor[T] {
	return Zip(a, b, func(x, y T) T { return x / y })
}

// Rem computes element-wise remainder of two tensors.
func Rem[T Integer](a, b *nune.Tensor[T]) *nune.Tensor[T] {
	return Zip(a, b, func(x, y T) T { return x % y })
}

// AddAssign computes element-wise addition of two tensors and assigns the result to the first tensor.
func AddAssign[T Number](a, b *nune.Tensor[T]) {
	ZipAssign(a, b, func(x, y T) T { return x + y })
}

// SubAssign computes element-wise subtraction of two tensors and assigns the result to the first tensor.
func SubAssign[T Number](a, b *nune.Tensor[T]) {
	ZipAssign(a, b, func(x, y T) T { return x - y })
}

// MulAssign computes element-wise multiplication of two tensors and assigns the result to the first tensor.
func MulAssign[T Number](a, b *nune.Tensor[T]) {
	ZipAssign(a, b, func(x, y T) T { return x * y })
}

// DivAssign computes element-wise division of two tensors and assigns the result to the first tensor.
func DivAssign[T Number](a, b *nune.Tensor[T]) {
	ZipAssign(a, b, func(x, y T) T { return x / y })
}

// RemAssign computes element-wise remainder of two tensors and assigns the result to the first tensor.
func RemAssign[T Integer](a, b *nune.Tensor[T]) {
	ZipAssign(a, b, func(x, y T) T { return x % y })
}

// handleZip is a helper function for Zip and ZipAssign.
// Copyright © The Nune Author. All rights reserved.
func handleZip[T, U, V Number](a *nune.Tensor[T], b *nune.Tensor[U], out *nune.Tensor[V], f func(T, U) V, nCPU int) {
	if a.Rank() == 0 {
		out.Ravel()[0] = f(a.Ravel()[0], b.Ravel()[0])
		return
	}
	var wg sync.WaitGroup

	for i := 0; i < nCPU; i++ {
		min := (i * a.Numel() / nCPU)
		max := ((i + 1) * a.Numel()) / nCPU

		wg.Add(1)
		go func(aBuf []T, bBuf []U, outBuf []V) {
			for j := 0; j < len(aBuf); j++ {
				outBuf[j] = f(aBuf[j], bBuf[j])
			}
			wg.Done()
		}(a.Ravel()[min:max], b.Ravel()[min:max], out.Ravel()[min:max])
	}
	wg.Wait()
	return
}
