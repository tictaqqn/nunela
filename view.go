package nunela

import (
	"github.com/vorduin/nune"
	"github.com/vorduin/slices"
)

func TryView[T Number](tensor *nune.Tensor[T], axis int, x int) (*nune.Tensor[T], error) {
	if axis >= tensor.Rank() || x >= tensor.Size(axis) {
		return nil, NewErrInappropriateAxisAndAxisNumber(tensor, axis, x)
	}
	data := make([]T, 0, tensor.Numel()/tensor.Size(axis))

	prod := slices.Prod(tensor.Shape()[axis+1:])
	if prod == 0 {
		prod = 1
	}
	prodOneMore := prod * tensor.Shape()[axis]
	for i, val := range tensor.Ravel() {
		if (i%prodOneMore)/prod == x {
			data = append(data, val)
		}
	}
	shape := make([]int, 0, tensor.Rank()-1)
	for i, size := range tensor.Shape() {
		if i != axis {
			shape = append(shape, size)
		}
	}
	arr := FromBufferWithShape(data, shape)
	return &arr, nil
}

// TODO: Implement TensorView without copying and methods similar to Tensor
func View[T Number](tensor *nune.Tensor[T], axis int, x int) *nune.Tensor[T] {
	out, err := TryView(tensor, axis, x)
	if err != nil {
		panic(err)
	}
	return out
}
