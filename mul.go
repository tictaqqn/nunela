package nunela

import "github.com/vorduin/nune"

func TryTensorDotWithOneAxis[T Number](tensors []*nune.Tensor[T], axes []int) (*nune.Tensor[T], error) {
	if len(tensors) != len(axes) {
		return nil, NewErrDifferentLen(tensors, axes)
	}
	if len(tensors) <= 1 {
		return nil, NewErrNotEnoughTensorsGiven()
	}
	lenAxis := tensors[0].Size(axes[0])
	for !every(tensors[1:], func(i int, t *nune.Tensor[T]) bool { return tensors[i].Size(axes[i]) == lenAxis }) {
		return nil, NewErrDifferentSizes(tensors...)
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
	for i := 0; i < lenAxis; i++ {
		tmp := nune.Ones[T](shape...)
		for j := range tensors {
			// TODO: remove .Ravel() after updating the behavior of nune.From
			tmp.Mul(View(tensors[j], axes[j], i).Ravel())
		}
		// TODO: remove .Ravel() after updating the behavior of nune.From
		sum.Add(tmp.Ravel())
	}
	return &sum, nil
}
