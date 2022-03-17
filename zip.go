package nunela

import (
	"sync"

	"github.com/vorduin/nune"
	"github.com/vorduin/slices"
)

func TryZip[T, U, V Number](a *nune.Tensor[T], b *nune.Tensor[U], f func(T, U) V) (*nune.Tensor[V], error) {
	if !slices.Equal(a.Shape(), b.Shape()) {
		return nil, NewErrDifferentShapesTwo(a, b)
	}
	out := nune.Zeros[V](a.Shape()...)
	handleZip(a, b, &out, f, configCPU(a.Numel()))
	return &out, nil
}

func TryZipAssign[T, U Number](a *nune.Tensor[T], b *nune.Tensor[U], f func(T, U) T) error {
	if !slices.Equal(a.Shape(), b.Shape()) {
		return NewErrDifferentShapesTwo(a, b)
	}
	handleZip(a, b, a, f, configCPU(a.Numel()))
	return nil
}

func Zip[T, U, V Number](a *nune.Tensor[T], b *nune.Tensor[U], f func(T, U) V) *nune.Tensor[V] {
	t, err := TryZip(a, b, f)
	if err != nil {
		panic(err)
	}
	return t
}

func ZipAssign[T, U Number](a *nune.Tensor[T], b *nune.Tensor[U], f func(T, U) T) {
	err := TryZipAssign(a, b, f)
	if err != nil {
		panic(err)
	}
}

func Add[T Number](a, b *nune.Tensor[T]) *nune.Tensor[T] {
	return Zip(a, b, func(x, y T) T { return x + y })
}

func Sub[T Number](a, b *nune.Tensor[T]) *nune.Tensor[T] {
	return Zip(a, b, func(x, y T) T { return x - y })
}

func Mul[T Number](a, b *nune.Tensor[T]) *nune.Tensor[T] {
	return Zip(a, b, func(x, y T) T { return x * y })
}

func Div[T Number](a, b *nune.Tensor[T]) *nune.Tensor[T] {
	return Zip(a, b, func(x, y T) T { return x / y })
}

func Rem[T Integer](a, b *nune.Tensor[T]) *nune.Tensor[T] {
	return Zip(a, b, func(x, y T) T { return x % y })
}

func AddAssign[T Number](a, b *nune.Tensor[T]) {
	ZipAssign(a, b, func(x, y T) T { return x + y })
}

func SubAssign[T Number](a, b *nune.Tensor[T]) {
	ZipAssign(a, b, func(x, y T) T { return x - y })
}

func MulAssign[T Number](a, b *nune.Tensor[T]) {
	ZipAssign(a, b, func(x, y T) T { return x * y })
}

func DivAssign[T Number](a, b *nune.Tensor[T]) {
	ZipAssign(a, b, func(x, y T) T { return x / y })
}

func RemAssign[T Integer](a, b *nune.Tensor[T]) {
	ZipAssign(a, b, func(x, y T) T { return x % y })
}

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
