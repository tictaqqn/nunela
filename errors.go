package nunela

import (
	"fmt"
	"strings"
)

func mapShapes[T Number](tensors ...Tensor[T]) []string {
	return mapSlice(tensors, func(tensor Tensor[T]) string { return fmt.Sprint(tensor.Shape()) })
}

func NewErrDifferentRanks[T Number](tensors ...Tensor[T]) error {
	return fmt.Errorf("nunela: received tensors of different ranks, %s", strings.Join(mapShapes(tensors...), ", "))
}

func NewErrDifferentRankAndIndices[T Number](tensor Tensor[T], indices []int) error {
	return fmt.Errorf("nunela: received indices that have length different from the rank of tensor, %d and %d", tensor.Rank(), len(indices))
}

func NewErrDifferentSizes[T Number](tensors ...Tensor[T]) error {
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
