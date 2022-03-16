package nunela

import (
	"fmt"
	"strings"

	"github.com/vorduin/nune"
)

func mapShapes[T Number](tensors ...*nune.Tensor[T]) []string {
	return mapSlice(tensors, func(tensor *nune.Tensor[T]) string { return fmt.Sprint(tensor.Shape()) })
}

func NewErrNotEnoughTensorsGiven() error {
	return fmt.Errorf("nunela: not enough number of tensors given")
}

func NewErrInappropriateAxisAndAxisNumber[T Number](tensor *nune.Tensor[T], axis int, x int) error {
	return fmt.Errorf("nunela: inappropriate axis %v and axis number %v for tensor with shape (%v)", axis, x, tensor.Shape())
}

func NewErrDifferentRanks[T Number](tensors ...*nune.Tensor[T]) error {
	return fmt.Errorf("nunela: received tensors of different ranks, %s", strings.Join(mapShapes(tensors...), ", "))
}

func NewErrDifferentRankAndIndices[T Number](tensor *nune.Tensor[T], indices []int) error {
	return fmt.Errorf("nunela: received indices that have length different from the rank of tensor, %d and %d", tensor.Rank(), len(indices))
}

func NewErrDifferentSizes[T Number](tensors ...*nune.Tensor[T]) error {
	return fmt.Errorf("nunela: received tensors of different sizes, %s", strings.Join(mapShapes(tensors...), ", "))
}

func NewErrInappropriateEinString(equation string) error {
	return fmt.Errorf("nunela: received equation string is inappropriate, or doesn't match the number of tensors lhs, %s", equation)
}

func NewErrDifferentLen[T any, U any](xs []T, ys []U) error {
	return fmt.Errorf("nunela: received slices of different lengths, %v and %v", xs, ys)
}

func NewErrDifferentIndices[T any](slices ...[]T) error {
	return fmt.Errorf("nunela: received slices of different indices, %v", strings.Join(mapSlice(slices, func(t []T) string { return fmt.Sprint(t) }), ", "))
}
