package nunela

import (
	"github.com/vorduin/nune"
)

func TryTensorDotWithOneAxis[T Number](tensors []*nune.Tensor[T], axes []int) (*nune.Tensor[T], error) {
	if len(tensors) != len(axes) {
		return nil, NewErrDifferentLen(tensors, axes)
	}
	if len(tensors) <= 1 {
		return nil, NewErrNotEnoughTensorsGiven()
	}
	lenAxis := tensors[0].Size(axes[0])
	for !every(tensors[1:], func(i int, t *nune.Tensor[T]) bool { return tensors[i].Size(axes[i]) == lenAxis }) {
		return nil, NewErrDifferentShapes(tensors...)
	}
	var shape []int
	for i := range tensors {
		for j, size := range tensors[i].Shape() {
			if j != axes[i] {
				shape = append(shape, size)
			}
		}
	}
	sum := nune.Zeros[T](shape...)
	prod := nune.Ones[T](shape...)
	r := rangeSlice(len(shape))
	for i := 0; i < lenAxis; i++ {
		rankRange := 0
		for j := range prod.Ravel() {
			prod.Ravel()[j] = T(1)
		}
		for j := range tensors {
			viewRank := tensors[j].Rank() - 1
			MulAssign(&prod, Repeat(View(tensors[j], axes[j], i), r[rankRange:rankRange+viewRank], shape))
			rankRange += viewRank
		}
		AddAssign(&sum, &prod)
	}
	return &sum, nil
}

func TensorDotWithOneAxis[T Number](tensors []*nune.Tensor[T], axes []int) *nune.Tensor[T] {
	out, err := TryTensorDotWithOneAxis(tensors, axes)
	if err != nil {
		panic(err)
	}
	return out
}
