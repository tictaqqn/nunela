package nunela

import "github.com/vorduin/nune"

func FromBufferWithShape[T Number](buf []T, shape []int) nune.Tensor[T] {
	// TODO: avoid copying buf with Reshape function
	return nune.FromBuffer(buf).Reshape(shape...)
}
