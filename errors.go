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

func NewErrInappropriateAxisAndAxisNumber[T Number](tensor *nune.Tensor[T], axis int, x [2]int) error {
	return fmt.Errorf("nunela: inappropriate axis %v and axis numbers %v for tensor with shape (%v)", axis, x, tensor.Shape())
}

func NewErrDifferentRanks[T Number](tensors ...*nune.Tensor[T]) error {
	return fmt.Errorf("nunela: received tensors of different ranks, %s", strings.Join(mapShapes(tensors...), ", "))
}

func NewErrDifferentRankAndIndices[T Number](tensor *nune.Tensor[T], indices []int) error {
	return fmt.Errorf("nunela: received indices that have length %d, different from the rank of tensor, %d", len(indices), tensor.Rank())
}

func NewErrDifferentRankAndAxisPairLen[T Number](tensor *nune.Tensor[T], axisPair map[int][2]int) error {
	return fmt.Errorf("nunela: received axis pairs that have length %d, different from the rank of tensor, %d", len(axisPair), tensor.Rank())
}

func NewErrInvalidAxisAndShape(baseAxis int, toShape []int) error {
	return fmt.Errorf("nunela: received invalid baseAxis %d for toShape %v", baseAxis, toShape)
}

func NewErrInvalidSizesWithBaseIndex(baseShape []int, toShape []int, baseAxis int) error {
	return fmt.Errorf("nunela: baseShape (%v) and toShape (%v) with baseAxis %d are incompatible", baseShape, toShape, baseAxis)
}

func NewErrDifferentShapes[T Number](tensors ...*nune.Tensor[T]) error {
	return fmt.Errorf("nunela: received tensors of different sizes, %s", strings.Join(mapShapes(tensors...), ", "))
}
func NewErrDifferentShapesTwo(tensors ...TensorLike) error {
	return fmt.Errorf("nunela: received tensors of different sizes, %s", strings.Join(mapSlice(tensors, func(t TensorLike) string { return fmt.Sprint(t.Shape()) }), ", "))
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
